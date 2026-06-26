from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import tqdm

from ..indexes.base import Base as BaseIndex
from ..profiling import NULL_PROFILER, use
from ..rank import RerankResult
from ..utils import iter_batch

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Common scaffolding for token-index retrievers.

    Subclasses implement :meth:`_score_batch` to convert a batch's token-level
    index hits into ranked ``RerankResult`` lists. The base owns input
    normalization, device defaulting, end-to-end short-circuit, and batching.
    """

    # Per-token candidates pulled from the index when the token path is used.
    default_k_token: int = 100
    default_batch_size: int = 50
    progress_desc: str = "Retrieving documents"

    def __init__(self, index: BaseIndex) -> None:
        self.index = index
        # Optional profiler (opt-in; the harness sets it). When None, all
        # span() calls go to NULL_PROFILER and cost ~nothing. After a retrieve,
        # ``last_profile`` holds this call's span trees (one per query batch),
        # or the end-to-end index's own tree on the e2e path.
        self.profiler = None
        self.last_profile = None

    def retrieve(
        self,
        queries_embeddings: list[list | np.ndarray | torch.Tensor],
        k: int = 10,
        k_token: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        subset: list[list[str]] | list[str] | None = None,
        maxsim_backend: str | None = None,
    ) -> list[list[RerankResult]]:
        """Retrieve documents for a list of queries.

        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of documents to retrieve.
        k_token
            The number of token-level candidates to retrieve from the index
            before scoring. Defaults to ``default_k_token`` for this retriever.
        device
            Device used for the scoring step. Defaults to the queries
            embeddings device when available, otherwise ``"cpu"``.
        batch_size
            Query batch size. Defaults to ``default_batch_size``.
        subset
            Optional document-id filter. End-to-end indexes receive it
            directly; on the token path, subclasses decide via
            :meth:`_validate_subset_token_path`.
        maxsim_backend
            MaxSim scoring kernel for the ColBERT token path
            (``"auto"`` / ``"torch"`` / ``"flash"`` / ``"lik"``); ``None`` defers
            to the ``PYLATE_SCORES_BACKEND`` env var / ``auto``. Ignored by
            retrievers that don't use ``colbert_scores`` (XTR) and by end-to-end
            indexes (which score internally).

        """
        k_token = self.default_k_token if k_token is None else k_token
        batch_size = self.default_batch_size if batch_size is None else batch_size

        # End-to-end indexes (e.g. PLAID, tachiom, fastplaid) handle scoring
        # internally and return RerankResult directly. We can't see their inner
        # stages from Python yet (that needs the rust instrumentation), but we
        # wrap the blocking call in one coarse ``search`` span so E2E indexes
        # flow through the same profiler/last_profile machinery — and so the
        # rust timing subtree has a parent to graft under once it exists.
        # device="cpu": the call blocks until results are materialised (the
        # returned ids/scores force any GPU work to complete), so wall-clock is
        # correct without a profiler-side CUDA sync.
        if self.index.is_end_to_end_index:
            kwargs = dict(queries_embeddings=queries_embeddings, k=k)
            if subset is not None:
                kwargs["subset"] = subset
            prof = self.profiler or NULL_PROFILER
            first_root = len(prof.roots)
            n = len(queries_embeddings) if hasattr(queries_embeddings, "__len__") else 1
            rust_roots_profile = None
            with use(prof):
                with prof.span("search", count=n, index=type(self.index).__name__) as sp:
                    results = self.index(**kwargs)
                    idx_prof = getattr(self.index, "last_profile", None)
                    if sp is not None and idx_prof:
                        rust_spans = idx_prof if isinstance(idx_prof, list) else [idx_prof]
                        if len(rust_spans) > 1 and all(
                            rust_span.name == "search" for rust_span in rust_spans
                        ):
                            rust_roots_profile = rust_spans
                        for rust_span in rust_spans:
                            if rust_span.name == "search" and rust_span.children:
                                sp.children.extend(rust_span.children)
                            else:
                                sp.children.append(rust_span)
            self.last_profile = rust_roots_profile or prof.roots[first_root:]
            return results

        self._validate_subset_token_path(subset)

        # Single-query input: a 2D array/tensor is one query of shape
        # (num_tokens, dim), not num_tokens queries — wrap in a list so the
        # batch loop sees a single element.
        if isinstance(queries_embeddings, (np.ndarray, torch.Tensor)):
            if queries_embeddings.ndim == 2:
                queries_embeddings = [queries_embeddings]

        if device is None:
            if queries_embeddings and isinstance(queries_embeddings[0], torch.Tensor):
                device = str(queries_embeddings[0].device)
            else:
                device = "cpu"

        if k > k_token:
            logger.warning(
                f"k ({k}) is greater than k_token ({k_token}), setting k_token to k."
            )
            k_token = k

        results: list[list[RerankResult]] = []
        progress_bar = tqdm.tqdm(
            iter_batch(queries_embeddings, batch_size=batch_size, tqdm_bar=False),
            desc=f"{self.progress_desc} (bs={batch_size})",
            disable=not self._show_progress(),
            total=math.ceil(len(queries_embeddings) / batch_size),
        )
        # Install the profiler as ambient so deep helpers (rerank, colbert_scores)
        # emit spans without it being threaded through their signatures. Each
        # batch produces one top-level "retrieve" span tree; collect this call's
        # trees into ``last_profile``. No-op overhead when profiler is None.
        prof = self.profiler or NULL_PROFILER
        first_root = len(prof.roots)
        with use(prof):
            for batch_queries_embeddings in progress_bar:
                n = len(batch_queries_embeddings)
                with prof.span("retrieve", count=n):
                    with prof.span("index_lookup", count=n):
                        hits = self.index(
                            queries_embeddings=batch_queries_embeddings,
                            k=k_token,
                        )
                    results.extend(
                        self._score_batch(
                            batch_queries_embeddings,
                            hits,
                            k=k,
                            device=device,
                            maxsim_backend=maxsim_backend,
                        )
                    )
        self.last_profile = prof.roots[first_root:]
        return results

    def _validate_subset_token_path(
        self, subset: list[list[str]] | list[str] | None
    ) -> None:
        """Hook for subclasses. Override to raise if ``subset`` cannot be
        honored on the token-path scoring branch. The default is a no-op
        (the argument is silently ignored on the token path)."""

    def _show_progress(self) -> bool:
        """Whether to render the per-batch tqdm bar. Defaults to ``True``;
        subclasses can override to expose a ``verbose`` knob."""
        return True

    @abstractmethod
    def _score_batch(
        self,
        batch_queries_embeddings: list | np.ndarray | torch.Tensor,
        hits: dict,
        *,
        k: int,
        device: str,
        maxsim_backend: str | None = None,
    ) -> list[list[RerankResult]]:
        """Convert one batch of index hits into ranked ``RerankResult`` lists.

        ``maxsim_backend`` selects the maxsim scoring kernel for the scoring path
        that uses it (ColBERT); retrievers that don't score with
        ``colbert_scores`` (e.g. XTR) accept and ignore it.
        """
