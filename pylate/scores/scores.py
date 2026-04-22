from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from ..utils.tensor import convert_to_tensor


def colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. The score is computed as the sum of maximum similarities
    between the query and the document.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)
    queries_mask
        The mask for the queries embeddings. Shape: (batch_size, num tokens queries)
    documents_mask
        The mask for the documents embeddings. Shape: (batch_size, num tokens documents)

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[10.], [0.], [1.]],
    ...     [[0.], [100.], [10.]],
    ...     [[1.], [0.], [1000.]],
    ... ])

    >>> documents_mask = torch.tensor([
    ...     [1., 1., 1.],
    ...     [1., 0., 1.],
    ...     [1., 1., 1.],
    ... ])
    >>> query_mask = torch.tensor([
    ...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
    ... ])

    >>> scores = colbert_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     queries_mask=query_mask,
    ...     documents_mask=documents_mask,
    ... )

    >>> scores
    tensor([[  10.,  10., 1000.],
            [  20.,  20., 2000.],
            [  0.,  0., 0.]])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    scores = torch.einsum(
        "ash,bth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(2)
    scores = scores.max(axis=-1).values.sum(axis=-1)
    return scores


def colbert_scores_pairwise(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Computes the ColBERT score for each query-document pair. The score is computed as the sum of maximum similarities
    between the query and the document for corresponding pairs.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[10.], [0.], [1.]],
    ...     [[0.], [100.], [1.]],
    ...     [[1.], [0.], [1000.]],
    ... ])

    >>> scores = colbert_scores_pairwise(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings
    ... )

    >>> scores
    tensor([  10.,  200., 3000.])

    """
    scores = []

    for query_embedding, document_embedding in zip(
        queries_embeddings, documents_embeddings
    ):
        query_embedding = convert_to_tensor(query_embedding)
        document_embedding = convert_to_tensor(document_embedding)

        query_document_score = torch.einsum(
            "sh,th->st",
            query_embedding,
            document_embedding,
        )

        scores.append(query_document_score.max(axis=-1).values.sum())

    return torch.stack(scores, dim=0)


def colbert_kd_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor = None,
    documents_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. This scoring function is dedicated to the knowledge distillation pipeline.

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[[10.], [0.], [1.]], [[20.], [0.], [1.]], [[30.], [0.], [1.]]],
    ...     [[[0.], [100.], [1.]], [[0.], [200.], [1.]], [[0.], [300.], [1.]]],
    ...     [[[1.], [0.], [1000.]], [[1.], [0.], [2000.]], [[10.], [0.], [3000.]]],
    ... ])
    >>> documents_mask = torch.tensor([
    ...     [[0., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ...     [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    ... ])
    >>> query_mask = torch.tensor([
    ...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
    ... ])
    >>> colbert_kd_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     queries_mask=query_mask,
    ...     documents_mask=documents_mask,
    ... )
    tensor([[ 1.,  20.,  30.],
            [200., 400., 600.],
            [  0.,   0.,   0.]])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    scores = torch.einsum(
        "ash,abth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)

    if documents_mask is not None:
        mask = convert_to_tensor(documents_mask)
        scores = scores * mask.unsqueeze(2)

    scores = scores.max(axis=-1).values.sum(axis=-1)
    return scores


# Adapted from PrimeQA (https://github.com/primeqa/primeqa branch:xtr)
# Specifically: https://github.com/primeqa/primeqa/blob/bb9385fa129a0dbb3c7aae96ad3c782913f8280d/primeqa/ir/dense/xtr_top/xtr/modeling/XTR.py

#   Copyright 2026 IBM PrimeQA Authors
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Changes:
# - extricated scoring function from E2E modeling class that also handled contrastive loss computation.
# - fixed a bug in the original implementation where the alignment mask was not being applied correctly.


class XTRScores:
    """XTR scoring using global top-k token retrieval.

    Each query's top-k token matches are selected globally across all Q*N documents in the
    batch, simulating retrieval from an index. Returns the full (Q, Q*N) cross-product score
    matrix so that all in-batch documents compete as negatives.

    Parameters
    ----------
    k
        Controls top-k token matching. Accepts:
        - ``int``: single k value, returns a single score tensor.
        - ``list[int]``: multiple k values with equal weights, returns
          ``list[tuple[Tensor, float]]``.
        - ``list[tuple[int, float]]``: multiple k values with explicit weights
          (normalized to sum to 1), returns ``list[tuple[Tensor, float]]``.

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1., 0.], [0., 0.]],
    ...     [[0., 1.], [0., 0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]],
    ...     [[[0., 1.], [1., 0.]], [[1., 0.], [0., 1.]]],
    ... ])

    >>> scores = XTRScores(k=2)(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ... )
    >>> scores.shape
    torch.Size([2, 4])

    Multi-k returns a list of (scores, weight) tuples:

    >>> result = XTRScores(k=[1, 2])(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ... )
    >>> len(result)
    2
    >>> result[0][0].shape
    torch.Size([2, 4])

    """

    requires_full_batch = True

    def __init__(self, k: int | list[int] | list[tuple[int, float]] = 128):
        if isinstance(k, int):
            self._k_weights: list[tuple[int, float]] | None = None
            self.k = k
        else:
            # list[int] or list[tuple[int, float]]
            if isinstance(k[0], int):
                w = 1.0 / len(k)
                self._k_weights = [(kv, w) for kv in k]
            else:
                total = sum(w for _, w in k)
                self._k_weights = [(kv, w / total) for kv, w in k]
            self.k = max(kv for kv, _ in self._k_weights)

    def compile(self, *args, **kwargs):
        self.__call__ = torch.compile(self.__call__, *args, **kwargs)

    def _score_for_k(
        self,
        clubbed: torch.Tensor,
        k: int,
        Qb: int,
        Db: int,
        Dt: int,
        queries_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute XTR scores from pre-computed clubbed token scores for a single k."""
        _, indices = clubbed.half().topk(k, dim=-1, sorted=False)
        mask = torch.zeros_like(clubbed, dtype=torch.bool).scatter_(-1, indices, True)
        masked = clubbed * mask
        topk_scores_max = masked.view(Qb, -1, Db, Dt).max(dim=-1).values

        if queries_mask is not None:
            topk_scores_max = topk_scores_max * queries_mask.unsqueeze(-1)

        scores_sum = topk_scores_max.sum(dim=1)
        Z = topk_scores_max.gt(0).float().sum(dim=1).clamp_(min=1e-3)
        return (scores_sum / Z).float()

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | list[tuple[torch.Tensor, float]]:
        queries_embeddings = convert_to_tensor(queries_embeddings)
        documents_embeddings = convert_to_tensor(documents_embeddings)

        Qb = queries_embeddings.shape[0]
        Dq, N = documents_embeddings.shape[:2]
        Db = Dq * N
        Qt = queries_embeddings.shape[1]
        Dt = documents_embeddings.shape[-2]
        H = queries_embeddings.shape[-1]

        docs_flat = documents_embeddings.view(Db, Dt, H)

        # Single large matmul — tensor core friendly
        Q_flat = queries_embeddings.reshape(Qb * Qt, H)
        D_flat = docs_flat.reshape(Db * Dt, H).T
        scores = (Q_flat @ D_flat).view(Qb, Qt, Db, Dt)

        if documents_mask is not None:
            docs_mask_flat = documents_mask.view(Db, Dt)
            scores = scores.masked_fill(
                ~docs_mask_flat.bool().unsqueeze(0).unsqueeze(0), -99999
            )

        clubbed = scores.flatten(2, 3)  # (Qb, Qt, Db*Dt)

        if self._k_weights is None:
            return self._score_for_k(clubbed, self.k, Qb, Db, Dt, queries_mask)

        return [
            (self._score_for_k(clubbed, k, Qb, Db, Dt, queries_mask), w)
            for k, w in self._k_weights
        ]


class ScopedBatchScores:
    """Full-batch scorer for contrastive losses with optional chunking.

    This scorer accepts a full query batch ``(B, Q, H)`` and a full 2D
    document batch ``(B, N, D, H)``.

    Shape convention:
    - ``B``: batch size
    - ``Q``: query sequence length
    - ``N``: n-way documents per query
    - ``D``: document sequence length
    - ``H``: hidden dimension

    It supports two scoring modes:
    - ``"colbert"``: token max-sim ColBERT scoring with chunking across both
      query batch, document batch, and N-way document dimensions.
    - ``"xtr"``: global top-k XTR scoring with query-batch chunking only.
      XTR requires every query chunk to see all documents simultaneously.

    It also supports:
    - ``scoring_scope``: whether scores are computed against local docs (in the same instance) per query
      or the full in-batch document pool.
    - ``return_scope``: whether to return local ``(B, N)`` scores or global
      ``(B, B*N)`` scores.
    """

    def __init__(
        self,
        mode: Literal["colbert", "xtr"] = "colbert",
        scoring_scope: Literal["global", "local"] = "global",
        return_scope: Literal["global", "local"] = "global",
        query_batch_chunk: int | None = None,
        doc_batch_chunk: int | None = None,
        doc_nway_chunk: int | None = None,
        xtr_k: int = 128,
    ) -> None:
        if mode not in {"colbert", "xtr"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'colbert' or 'xtr'.")

        if query_batch_chunk is not None and query_batch_chunk <= 0:
            raise ValueError("query_batch_chunk must be > 0 when provided.")

        if doc_batch_chunk is not None and doc_batch_chunk <= 0:
            raise ValueError("doc_batch_chunk must be > 0 when provided.")

        if doc_nway_chunk is not None and doc_nway_chunk <= 0:
            raise ValueError("doc_nway_chunk must be > 0 when provided.")

        self.mode = mode
        self.scoring_scope = scoring_scope
        self.return_scope = return_scope
        self.query_batch_chunk = query_batch_chunk
        self.doc_batch_chunk = doc_batch_chunk
        self.doc_nway_chunk = doc_nway_chunk
        self._xtr_k = xtr_k

        if self.scoring_scope not in {"global", "local"}:
            raise ValueError(
                f"Unsupported scoring_scope: {self.scoring_scope}. Expected 'global' or 'local'."
            )
        if self.return_scope not in {"global", "local"}:
            raise ValueError(
                f"Unsupported return_scope: {self.return_scope}. Expected 'global' or 'local'."
            )
        if self.mode == "xtr" and self.scoring_scope != "global":
            raise ValueError("XTR requires scoring_scope='global'.")
        if self.mode == "xtr" and self.doc_batch_chunk is not None:
            raise ValueError(
                "doc_batch_chunk is only supported for mode='colbert'. "
                "XTR requires scoring against the full document batch."
            )
        if self.mode == "xtr" and self.doc_nway_chunk is not None:
            raise ValueError(
                "doc_nway_chunk is only supported for mode='colbert'. "
                "XTR requires scoring against all N-way documents at once."
            )
        if self.return_scope == "global" and self.scoring_scope == "local":
            raise ValueError(
                "return_scope='global' requires scoring_scope='global'."
            )
        if self.scoring_scope == "local" and self.doc_batch_chunk is not None:
            raise ValueError(
                "doc_batch_chunk is not supported for scoring_scope='local'."
            )

    @staticmethod
    def _validate_documents_shape(documents_embeddings: torch.Tensor) -> None:
        if documents_embeddings.ndim != 4:
            raise ValueError(
                "documents_embeddings must be 4D with shape (B, N, D, H) "
                f"for full-batch scoring, got shape {tuple(documents_embeddings.shape)}."
            )

    @staticmethod
    def _compute_token_scores(
        queries_embeddings: torch.Tensor,
        documents_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        B = queries_embeddings.shape[0]
        Q = queries_embeddings.shape[1]
        B_docs = documents_embeddings.shape[0]
        D = documents_embeddings.shape[1]
        H = queries_embeddings.shape[-1]

        q_flat = queries_embeddings.reshape(B * Q, H)
        d_flat = documents_embeddings.reshape(B_docs * D, H).T
        return (q_flat @ d_flat).view(B, Q, B_docs, D)

    @staticmethod
    def _reduce_scores(
        token_scores: torch.Tensor,
        queries_mask: torch.Tensor | None,
        documents_mask: torch.Tensor | None,
        xtr_k: int | None = None,
    ) -> torch.Tensor:
        """Reduce token-level similarities to document scores.

        Shared path:
        - optional document masking
        - max over document tokens
        - optional query masking
        - sum over query tokens

        XTR adds global top-k selection and Z-normalization.
        """
        B, Q, B_docs, D = token_scores.shape

        if xtr_k is None:
            # Independent-document path uses multiplicative document masking.
            if documents_mask is not None:
                token_scores = token_scores * documents_mask.unsqueeze(0).unsqueeze(0)
        else:
            if documents_mask is not None:
                token_scores = token_scores.masked_fill(
                    ~documents_mask.bool().unsqueeze(0).unsqueeze(0),
                    -99999,
                )
            clubbed = token_scores.flatten(2, 3)  # (B, Q, B_docs*D)
            _, indices = clubbed.half().topk(xtr_k, dim=-1, sorted=False)
            topk_mask = torch.zeros_like(clubbed, dtype=torch.bool).scatter_(
                -1, indices, True
            )
            token_scores = (clubbed * topk_mask).view(B, Q, B_docs, D)

        max_scores = token_scores.max(dim=-1).values  # (B, Q, B_docs)
        if queries_mask is not None:
            max_scores = max_scores * queries_mask.unsqueeze(-1)

        scores_sum = max_scores.sum(dim=1)
        if xtr_k is None:
            return scores_sum

        Z = max_scores.gt(0).float().sum(dim=1).clamp_(min=1e-3)
        return (scores_sum / Z).float()

    def _score_chunk(
        self,
        queries_embeddings: torch.Tensor,
        documents_embeddings: torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a single (query chunk, doc chunk) pair.

        `documents_embeddings` must be shaped `(B_docs, N_chunk, D, H)` for this
        chunk. Returns a score block shaped `(B_query, B_docs, N_chunk)`.
        """
        B_docs, N_chunk = documents_embeddings.shape[:2]
        docs_flat = documents_embeddings.reshape(
            B_docs * N_chunk,
            documents_embeddings.shape[-2],
            documents_embeddings.shape[-1],
        )
        docs_mask_flat = (
            None
            if documents_mask is None
            else documents_mask.reshape(B_docs * N_chunk, documents_mask.shape[-1])
        )
        xtr_k = self._xtr_k if self.mode == "xtr" else None
        token_scores = self._compute_token_scores(queries_embeddings, docs_flat)
        return self._reduce_scores(
            token_scores=token_scores,
            queries_mask=queries_mask,
            documents_mask=docs_mask_flat,
            xtr_k=xtr_k,
        ).view(queries_embeddings.shape[0], B_docs, N_chunk)

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
        query_start_index: int = 0,
    ) -> torch.Tensor:
        queries_embeddings = convert_to_tensor(queries_embeddings)
        documents_embeddings = convert_to_tensor(documents_embeddings)
        self._validate_documents_shape(documents_embeddings)

        if query_start_index < 0:
            raise ValueError("query_start_index must be >= 0.")

        B = queries_embeddings.shape[0]
        query_step = self.query_batch_chunk or B
        B_docs_total, N = documents_embeddings.shape[:2]

        is_xtr = self.mode == "xtr"
        local_scoring = self.scoring_scope == "local"
        local_return = self.return_scope == "local"

        doc_batch_step_global = (
            B_docs_total if is_xtr else (self.doc_batch_chunk or B_docs_total)
        )
        doc_nway_step_global = N if is_xtr else (self.doc_nway_chunk or N)
        reduced_score_chunks = []
        for b_start in range(0, B, query_step):
            b_end = min(b_start + query_step, B)
            b_query = b_end - b_start
            abs_b_start = query_start_index + b_start
            abs_b_end = query_start_index + b_end
            q_chunk_mask = (
                None if queries_mask is None else queries_mask[b_start:b_end]
            )

            if local_scoring:
                if abs_b_end > B_docs_total:
                    raise ValueError(
                        "scoring_scope='local' requires matching document rows for each "
                        "query in the current chunk."
                    )
                docs_for_chunk = documents_embeddings[abs_b_start:abs_b_end]
                docs_mask_for_chunk = (
                    None
                    if documents_mask is None
                    else documents_mask[abs_b_start:abs_b_end]
                )
                doc_batch_step = b_query
                doc_nway_step = self.doc_nway_chunk or N
            else:
                docs_for_chunk = documents_embeddings
                docs_mask_for_chunk = documents_mask
                doc_batch_step = doc_batch_step_global
                doc_nway_step = doc_nway_step_global

            q_chunk_scores = torch.empty(
                b_query,
                docs_for_chunk.shape[0],
                N,
                device=queries_embeddings.device,
                dtype=queries_embeddings.dtype,
            )
            for d_start in range(0, docs_for_chunk.shape[0], doc_batch_step):
                d_end = min(d_start + doc_batch_step, docs_for_chunk.shape[0])
                for n_start in range(0, N, doc_nway_step):
                    n_end = min(n_start + doc_nway_step, N)
                    docs_chunk = docs_for_chunk[d_start:d_end, n_start:n_end]
                    docs_mask_chunk = (
                        None
                        if docs_mask_for_chunk is None
                        else docs_mask_for_chunk[d_start:d_end, n_start:n_end]
                    )
                    q_chunk_scores[:, d_start:d_end, n_start:n_end] = self._score_chunk(
                        queries_embeddings=queries_embeddings[b_start:b_end],
                        documents_embeddings=docs_chunk,
                        queries_mask=q_chunk_mask,
                        documents_mask=docs_mask_chunk,
                    )

            if local_return:
                if local_scoring:
                    local_idx = torch.arange(
                        b_query, device=queries_embeddings.device
                    )
                    reduced_score_chunks.append(q_chunk_scores[local_idx, local_idx, :])
                else:
                    local_cols = (
                        torch.arange(
                            abs_b_start, abs_b_end, device=queries_embeddings.device
                        ).unsqueeze(1)
                        * N
                    ) + torch.arange(N, device=queries_embeddings.device)
                    reduced_score_chunks.append(
                        q_chunk_scores.view(b_query, B_docs_total * N).gather(
                            1, local_cols
                        )
                    )
            else:
                reduced_score_chunks.append(
                    q_chunk_scores.view(b_query, B_docs_total * N)
                )

        return torch.cat(reduced_score_chunks, dim=0)


class XTRKDScores:
    """XTR scores for knowledge distillation. Same global top-k scoring as
    :class:`XTRScores`, but returns only each query's own N-way document scores
    to match the ``(Q, N)`` interface expected by :class:`~pylate.losses.Distillation`.

    Parameters
    ----------
    k
        Number of top token matches to retain per query token.
    """

    requires_full_batch = True

    def __init__(self, k: int = 128):
        self._xtr_scores = XTRScores(k=k)

    @property
    def k(self):
        return self._xtr_scores.k

    @k.setter
    def k(self, value):
        self._xtr_scores.k = value

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        documents_embeddings = convert_to_tensor(documents_embeddings)
        Q, N = documents_embeddings.shape[:2]

        # Full cross-product scores: (Q, Q*N)
        all_scores = self._xtr_scores(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
        )

        # Slice out each query's own N documents
        def _slice(scores: torch.Tensor) -> torch.Tensor:
            idx = torch.arange(Q, device=scores.device).unsqueeze(1) * N + torch.arange(
                N, device=scores.device
            )
            return scores.gather(1, idx)

        if isinstance(all_scores, list):
            return [(_slice(s), w) for s, w in all_scores]
        return _slice(all_scores)
