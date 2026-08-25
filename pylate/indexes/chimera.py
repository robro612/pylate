from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from ..profiling import Span, active
from ..rank import RerankResult
from .base import Base

logger = logging.getLogger(__name__)

# PADDED_DIM and Q_DOCLEN are `#define`s in Chimera's config.cuh, baked into the
# binary and not exported through the bindings. The wrapper needs them to reject
# a mismatched corpus with a useful message instead of letting
# std::invalid_argument surface from inside a search.
_DEFAULT_PADDED_DIM = 128
_DEFAULT_QUERY_TOKENS = 32
_MIN_DIM = 64  # build_index.cu rejects dimension < 64
# nprobe becomes CAGRA's search k and cagra_itopk_size its intermediate top-k;
# RAFT's topk kernel hard-fails above 1024 ("topk must be lower than or equal
# to 1024"), several seconds into a search rather than at construction.
_CAGRA_MAX_TOPK = 1024

_CHIMERA_MODULE = None


def _load_chimera():
    """Import the compiled ``chimera`` extension, caching the result.

    Raises with the environment command rather than a bare ImportError: the
    module is missing far more often because this venv was not built with the
    extra than because anything is wrong.
    """
    global _CHIMERA_MODULE
    if _CHIMERA_MODULE is None:
        try:
            import chimera
        except ImportError as exc:
            raise ImportError(
                "The compiled Chimera extension is not installed in this "
                "environment. It is the `chimera` extra, built from "
                "third_party/Chimera by:\n"
                "    srunl40s --mem=64G --cpus-per-task=16 "
                "./scripts/sync_env.sh cu130\n"
                "cu126 does not have it: Chimera needs AVX-512, and the V100 "
                "nodes are Broadwell Xeons without it."
            ) from exc
        _CHIMERA_MODULE = chimera
    return _CHIMERA_MODULE


def _compile_time_limits() -> tuple[int, int]:
    """Return ``(padded_dim, query_tokens)`` for the installed build.

    Read off the module, which exports the `#define`s it was compiled with, so
    an environment built for a longer query length is detected rather than
    assumed. The env vars are an escape hatch for a build predating that
    export; the constants themselves fall back to upstream's ColBERTv2 shape.
    """
    module = _load_chimera()
    return (
        int(
            os.environ.get(
                "CHIMERA_PADDED_DIM", getattr(module, "PADDED_DIM", _DEFAULT_PADDED_DIM)
            )
        ),
        int(
            os.environ.get(
                "CHIMERA_Q_DOCLEN", getattr(module, "Q_DOCLEN", _DEFAULT_QUERY_TOKENS)
            )
        ),
    )


class Chimera(Base):
    """Chimera index — single-GPU multi-vector retrieval (https://github.com/iidyc/Chimera).

    Wraps the C++/CUDA `ChimeraIndex` pybind11 class as an end-to-end PyLate
    index, so it can be benchmarked head-to-head with ``PLAID``/``FastPlaid``
    and ``TachiomIndex`` through the same retriever and harness.

    Chimera clusters document tokens with RaBitQ-style residual quantisation,
    probes clusters through a CAGRA graph over the centroids, and reranks a
    candidate pool with full-bit codes on the CPU (AVX-512). Its build and
    search knobs map roughly onto PLAID's: ``n_clusters`` ~ centroid count,
    ``ex_bits`` ~ ``nbits``, ``nprobe`` ~ ``n_ivf_probe``, ``k_refine`` /
    ``k_full_bit`` ~ ``n_full_scores``.

    Four properties of the upstream backend leak into this wrapper and are
    worth knowing before reading a benchmark number:

    * **Scores are ranks, not similarities.** The pybind11 `search` returns
      document ids only; the late-interaction scores exist inside
      ``collaborative_document_scoring`` but are dropped when its result heap is
      drained. Documents come back in descending-score order, so this class
      synthesises strictly decreasing placeholder scores. Rank metrics (NDCG,
      recall, MAP, MRR) are exact; anything that reads score *values* — score
      distributions, fusion, calibration — is not. ``scores_are_synthetic``
      says which of the two you got, and the wrapper uses real scores
      automatically if the loaded build exposes a scored search.
    * **Query length and dimension are compile-time constants.** ``Q_DOCLEN``
      (32) and ``PADDED_DIM`` (128) come from ``config.cuh``. Shorter queries
      are zero-padded, which is score-preserving for MaxSim — a zero query
      vector contributes 0 to every document — though it still spends its share
      of the candidate budget on whichever clusters it probes. Verified: a
      31-token query padded to 32 ranks its own document first on 100/100
      repeats. Longer queries raise rather than truncate, since silently
      dropping tokens would change what the benchmark measures.
    * **The index is reloaded from disk after building** — searching the object
      ``ChimeraIndex.build`` returns drops the correct answer about 10% of the
      time. See ``_build`` for the measurement.
    * **Building is not incremental.** ``ChimeraIndex.build`` takes the whole
      corpus at once and copies the float32 array into a ``std::vector``, so
      peak host memory is about ``2 * n_tokens * dim * 4`` bytes on top of the
      shards. There is no update or delete path.

    Parameters
    ----------
    index_folder
        The folder where the index will be stored.
    index_name
        The name of the index.
    override
        Whether to delete an existing index of the same name first.
    n_clusters
        Number of coarse clusters. ``None`` derives it from the corpus as
        ``n_tokens // tokens_per_cluster``; upstream recommends roughly
        ``n_tokens / 200`` to ``n_tokens / 100``.
    tokens_per_cluster
        Divisor used when ``n_clusters`` is ``None``.
    ex_bits
        Residual quantisation width, 1-7. Total is ``1 + ex_bits`` bits per
        dimension, so ``ex_bits=4`` is a little denser than PLAID's ``nbits=4``.
    nprobe
        Clusters probed per query token. Must not exceed ``n_clusters``.
    k_refine
        Candidate documents kept for refinement.
    k_full_bit
        Refined candidates rescored with full-bit codes. Must not exceed
        ``k_refine``.
    cagra_itopk_size
        Intermediate CAGRA top-k. ``None`` picks a value slightly above
        ``nprobe``, which is what upstream recommends.
    num_chunks
        Chunks used by collaborative document scoring — the pipelining depth
        between GPU one-bit scoring and CPU full-bit rescoring.
    """

    is_end_to_end_index = True

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "chimera",
        override: bool = False,
        n_clusters: int | None = None,
        tokens_per_cluster: int = 150,
        ex_bits: int = 4,
        nprobe: int = 128,
        k_refine: int = 3000,
        k_full_bit: int = 300,
        cagra_itopk_size: int | None = None,
        num_chunks: int = 5,
    ) -> None:
        self._chimera = _load_chimera()
        self.padded_dim, self.query_tokens = _compile_time_limits()

        if not 1 <= ex_bits <= 7:
            raise ValueError(f"ex_bits must be in [1, 7], got {ex_bits}")
        if k_full_bit > k_refine:
            raise ValueError(
                f"k_full_bit ({k_full_bit}) must not exceed k_refine ({k_refine})"
            )
        if nprobe > _CAGRA_MAX_TOPK:
            raise ValueError(
                f"nprobe ({nprobe}) exceeds {_CAGRA_MAX_TOPK}. Chimera probes "
                "clusters with a CAGRA search whose k is nprobe, and RAFT's topk "
                "kernel rejects k > 1024. Raise k_refine / k_full_bit instead — "
                "on nfcorpus, k_full_bit is the knob that recovers recall."
            )

        self.index_folder = index_folder
        self.index_name = index_name
        self.n_clusters = n_clusters
        self.tokens_per_cluster = tokens_per_cluster
        self.ex_bits = ex_bits
        self.nprobe = nprobe
        self.k_refine = k_refine
        self.k_full_bit = k_full_bit
        # Upstream advises "slightly larger than nprobe"; the CAGRA ceiling
        # applies here too, so it cannot simply track nprobe upward.
        self.cagra_itopk_size = (
            min(max(64, nprobe + 32), _CAGRA_MAX_TOPK)
            if cagra_itopk_size is None
            else cagra_itopk_size
        )
        if self.cagra_itopk_size > _CAGRA_MAX_TOPK:
            raise ValueError(
                f"cagra_itopk_size ({self.cagra_itopk_size}) exceeds "
                f"{_CAGRA_MAX_TOPK}, RAFT's topk kernel limit."
            )
        self.num_chunks = num_chunks

        self.index_path = os.path.join(index_folder, index_name)
        if override and os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path, exist_ok=True)

        self._chimera_path = os.path.join(self.index_path, "chimera_index")
        self._params_path = os.path.join(self.index_path, "params.json")
        self._doc_id_to_int_path = os.path.join(self.index_path, "doc_id_to_int.pkl")
        self._int_to_doc_id_path = os.path.join(self.index_path, "int_to_doc_id.pkl")

        self._index = None
        self._doc_id_to_int: dict | None = None
        self._int_to_doc_id: dict | None = None
        self._params: dict = {}
        self.last_profile = None

        # `search` returns ids only on stock upstream. A build that also returns
        # scores exposes one of these; probing by name keeps this wrapper working
        # against both without a version check.
        self._scored_search = next(
            (
                name
                for name in ("search_scored", "search_with_scores")
                if hasattr(self._chimera.ChimeraIndex, name)
            ),
            None,
        )
        self.scores_are_synthetic = self._scored_search is None

        self.is_indexed = os.path.exists(self._params_path)
        if self.is_indexed:
            with open(self._params_path) as f:
                self._params = json.load(f)
            # Search parameters are bound at load time by the C++ index and
            # cannot be changed afterwards, so the constructor's values are what
            # this instance will search with — but the build-time ones have to
            # come off disk, since PQ layout and cluster count are properties of
            # the artifact, not of this call.
            self.n_clusters = self._params.get("n_clusters", self.n_clusters)
            self.ex_bits = self._params.get("ex_bits", self.ex_bits)

    # -- properties ---------------------------------------------------------

    @property
    def actual_total_centroids(self) -> int | None:
        """Coarse-cluster count the built/loaded index holds.

        Named to match ``TachiomIndex`` so the benchmark harness logs it for
        Chimera without a special case.
        """
        return self._params.get("n_clusters")

    @property
    def dim(self) -> int | None:
        return self._params.get("dim")

    @property
    def n_tokens(self) -> int | None:
        return self._params.get("n_tokens")

    # -- persistence --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        if not self.is_indexed:
            raise ValueError(
                "The index is empty. Please add documents before querying."
            )
        built_tokens = self._params.get("query_tokens")
        built_dim = self._params.get("padded_dim")
        if (built_tokens, built_dim) != (None, None) and (
            built_tokens != self.query_tokens or built_dim != self.padded_dim
        ):
            raise RuntimeError(
                f"This index was built by a Chimera compiled for "
                f"Q_DOCLEN={built_tokens} / PADDED_DIM={built_dim}, but the "
                f"installed extension is {self.query_tokens} / "
                f"{self.padded_dim}. The rotation and code layout depend on "
                "both, so the index has to be rebuilt (or the extension "
                "rebuilt to match)."
            )
        n_clusters = self._params.get("n_clusters")
        if n_clusters is not None and self.nprobe > n_clusters:
            warnings.warn(
                f"nprobe ({self.nprobe}) exceeds the index's cluster count "
                f"({n_clusters}); clamping. Chimera rejects nprobe > n_clusters.",
                UserWarning,
                stacklevel=2,
            )
            self.nprobe = n_clusters
            self.cagra_itopk_size = min(
                max(self.cagra_itopk_size, self.nprobe), _CAGRA_MAX_TOPK
            )
        self._index = self._chimera.ChimeraIndex.load(
            self._chimera_path, **self._search_options()
        )

    def _search_options(self) -> dict:
        return dict(
            nprobe=self.nprobe,
            k_refine=self.k_refine,
            k_full_bit=self.k_full_bit,
            cagra_itopk_size=self.cagra_itopk_size,
            num_chunks=self.num_chunks,
        )

    def _ensure_mappings(self) -> None:
        if self._int_to_doc_id is None:
            if not os.path.exists(self._int_to_doc_id_path):
                raise FileNotFoundError(
                    f"Document ID mapping not found at {self._int_to_doc_id_path}. "
                    "Please call add_documents before querying."
                )
            with open(self._int_to_doc_id_path, "rb") as f:
                self._int_to_doc_id = pickle.load(f)
        if self._doc_id_to_int is None and os.path.exists(self._doc_id_to_int_path):
            with open(self._doc_id_to_int_path, "rb") as f:
                self._doc_id_to_int = pickle.load(f)

    def _save_mappings(self, doc_id_to_int: dict, int_to_doc_id: dict) -> None:
        with open(self._doc_id_to_int_path, "wb") as f:
            pickle.dump(doc_id_to_int, f)
        with open(self._int_to_doc_id_path, "wb") as f:
            pickle.dump(int_to_doc_id, f)
        self._doc_id_to_int = doc_id_to_int
        self._int_to_doc_id = int_to_doc_id

    # -- building -----------------------------------------------------------

    def _resolve_n_clusters(self, n_tokens: int) -> int:
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = max(1, n_tokens // max(self.tokens_per_cluster, 1))
        n_clusters = max(1, min(int(n_clusters), n_tokens))
        if self.nprobe > n_clusters:
            warnings.warn(
                f"nprobe ({self.nprobe}) exceeds n_clusters ({n_clusters}); "
                "clamping to n_clusters.",
                UserWarning,
                stacklevel=2,
            )
            self.nprobe = n_clusters
            self.cagra_itopk_size = min(
                max(self.cagra_itopk_size, self.nprobe), _CAGRA_MAX_TOPK
            )
        return n_clusters

    def _build(
        self,
        embeddings: np.ndarray,
        doc_lens: np.ndarray,
        documents_ids: list[str],
    ) -> "Chimera":
        """Build, save, and record the artifact. ``embeddings`` must be C-contiguous f32."""
        n_tokens, dim = embeddings.shape
        if not _MIN_DIM <= dim <= self.padded_dim:
            raise ValueError(
                f"Chimera accepts embedding dimensions in "
                f"[{_MIN_DIM}, {self.padded_dim}]; got {dim}. PADDED_DIM is a "
                "compile-time constant — edit cpp/include/chimera/config.cuh "
                "(a positive multiple of 64) and rebuild to change it."
            )
        n_clusters = self._resolve_n_clusters(n_tokens)

        logger.info(
            "Chimera.build: %d docs, %d tokens, dim=%d, n_clusters=%d, ex_bits=%d",
            len(doc_lens), n_tokens, dim, n_clusters, self.ex_bits,
        )
        build_start = time.perf_counter()
        self._index = self._chimera.ChimeraIndex.build(
            embeddings,
            doc_lens.astype(np.int32).tolist(),
            n_clusters=n_clusters,
            ex_bits=self.ex_bits,
            **self._search_options(),
        )
        build_seconds = time.perf_counter() - build_start

        # chimera_index::save writes into an existing directory; it does not
        # create one.
        os.makedirs(self._chimera_path, exist_ok=True)
        self._index.save(self._chimera_path)

        # Drop the built object and let _ensure_loaded() read the artifact back.
        # This is not tidiness: searching the index that build() returns is
        # unreliable upstream. Measured on 2000 self-queries (a document's own
        # tokens as the query, which must rank it first), same process, same
        # search options, identical contents:
        #
        #     build()-returned : 5/5 queries gave >1 distinct top-10 across
        #                        repeats; the query's own document missed top-1
        #                        in 54/500 calls
        #     load()-ed        : 0/5 unstable, 0/500 misses
        #
        # The data written by save() is correct; only the in-memory post-build
        # search state is not, so the round-trip through disk costs a reload
        # and buys determinism. Revisit if upstream fixes
        # initialize_search_state() on the build path.
        self._index = None

        self._params = {
            "n_clusters": n_clusters,
            "ex_bits": self.ex_bits,
            "dim": int(dim),
            "n_tokens": int(n_tokens),
            "n_documents": int(len(doc_lens)),
            "padded_dim": self.padded_dim,
            "query_tokens": self.query_tokens,
            "build_seconds": round(build_seconds, 3),
        }
        with open(self._params_path, "w") as f:
            json.dump(self._params, f, indent=2)

        self._save_mappings(
            {doc_id: i for i, doc_id in enumerate(documents_ids)},
            dict(enumerate(documents_ids)),
        )
        self.n_clusters = n_clusters
        self.is_indexed = True
        return self

    def add_documents(
        self,
        documents_ids: str | list[str],
        documents_embeddings: list[np.ndarray | torch.Tensor],
        **kwargs,
    ) -> "Chimera":
        """Build the index from in-memory per-document token embeddings.

        Chimera has no incremental path: a second call raises rather than
        silently discarding the first corpus.
        """
        if isinstance(documents_ids, str):
            documents_ids = [documents_ids]
        documents_ids = list(documents_ids)

        if self.is_indexed:
            raise ValueError(
                "Chimera indexes are built in one shot and cannot be extended. "
                "Pass override=True to rebuild from scratch."
            )

        arrays = []
        for embedding in documents_embeddings:
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().detach().numpy()
            arrays.append(np.asarray(embedding, dtype=np.float32))
        if not arrays:
            raise ValueError("No document embeddings were provided.")

        doc_lens = np.array([a.shape[0] for a in arrays], dtype=np.int32)
        embeddings = np.ascontiguousarray(np.vstack(arrays), dtype=np.float32)
        del arrays
        return self._build(embeddings, doc_lens, documents_ids)

    def add_documents_from_shards(
        self,
        documents_ids: list[str],
        shard_dir: str,
        glob_pattern: str = "embeddings_*.npy",
    ) -> "Chimera":
        """Build the index from embedding shards on disk.

        Fills one pre-allocated float32 buffer shard by shard rather than
        materialising a list of per-document arrays. That is still the whole
        corpus in host RAM, and ``ChimeraIndex.build`` copies it again into a
        ``std::vector<float>`` before clustering, so budget roughly
        ``2 * n_tokens * dim * 4`` bytes.

        Shards follow the convention used by the rest of the repo:
        ``<stem>.npy`` alongside ``<stem>.doclens.npy``. Token IDs, when
        present, are ignored — Chimera has no vocabulary-aware clustering.
        """
        if self.is_indexed:
            raise ValueError(
                "Chimera indexes are built in one shot and cannot be extended. "
                "Pass override=True to rebuild from scratch."
            )

        shard_dir = Path(shard_dir)
        vec_paths = [
            p
            for p in sorted(shard_dir.glob(glob_pattern))
            if not any(
                p.name.endswith(s)
                for s in (".doclens.npy", ".doc_ids.npy", ".token_ids.npy")
            )
        ]
        if not vec_paths:
            raise FileNotFoundError(f"No embedding shards found in {shard_dir}")

        doclens_paths = [
            p.parent / p.name.replace(".npy", ".doclens.npy") for p in vec_paths
        ]
        all_doclens = [np.load(str(p)) for p in doclens_paths]
        shard_tok_counts = [int(d.sum()) for d in all_doclens]
        total_tokens = sum(shard_tok_counts)
        dim = int(np.load(str(vec_paths[0]), mmap_mode="r").shape[1])

        # float32 up front: the binding is declared `.noconvert()`, so a float16
        # buffer would be rejected outright rather than cast.
        embeddings = np.empty((total_tokens, dim), dtype=np.float32)
        doc_lens = np.concatenate(all_doclens).astype(np.int32)

        offset = 0
        for path, count in zip(vec_paths, shard_tok_counts):
            embeddings[offset : offset + count] = np.load(str(path), mmap_mode="r")
            offset += count

        if len(doc_lens) != len(documents_ids):
            raise ValueError(
                f"Shards describe {len(doc_lens)} documents but "
                f"{len(documents_ids)} document IDs were given."
            )
        return self._build(embeddings, doc_lens, list(documents_ids))

    # -- search -------------------------------------------------------------

    def _pack_queries(self, queries_embeddings) -> np.ndarray:
        """Return a C-contiguous float32 ``[n_queries, Q_DOCLEN, dim]`` batch."""
        if isinstance(queries_embeddings, torch.Tensor):
            queries_embeddings = queries_embeddings.cpu().detach().numpy()

        if isinstance(queries_embeddings, np.ndarray):
            arr = np.asarray(queries_embeddings, dtype=np.float32)
            queries = [arr] if arr.ndim == 2 else list(arr)
        else:
            queries = []
            for query in queries_embeddings:
                if isinstance(query, torch.Tensor):
                    query = query.cpu().detach().numpy()
                queries.append(np.asarray(query, dtype=np.float32))

        # PyLate's encode() hands back (1, Q, dim) per query when padding is on.
        queries = [
            q.squeeze(0) if q.ndim == 3 and q.shape[0] == 1 else q for q in queries
        ]

        dim = self._params.get("dim", queries[0].shape[1])
        packed = np.zeros(
            (len(queries), self.query_tokens, dim), dtype=np.float32
        )
        for i, query in enumerate(queries):
            n_tokens, query_dim = query.shape
            if query_dim != dim:
                raise ValueError(
                    f"Query dimension {query_dim} does not match the indexed "
                    f"dimension {dim}."
                )
            if n_tokens > self.query_tokens:
                raise ValueError(
                    f"Chimera was compiled for Q_DOCLEN={self.query_tokens} query "
                    f"tokens but a query has {n_tokens}. Truncating would change "
                    "what is being measured, so this raises instead: either lower "
                    "this dataset's `query_length` in conf/eval/config.yaml, or "
                    "raise Q_DOCLEN in cpp/include/chimera/config.cuh (a multiple "
                    "of 16) and rebuild."
                )
            # Shorter queries keep the zero rows: a zero query vector scores 0
            # against every document, so the MaxSim sum is unchanged.
            packed[i, :n_tokens] = query
        return np.ascontiguousarray(packed)

    def __call__(
        self,
        queries_embeddings: (
            np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor]
        ),
        k: int = 10,
    ) -> list[list[RerankResult]]:
        """Search the index for the nearest documents to each query.

        Parameters
        ----------
        queries_embeddings
            Query token embeddings: a list of 2D arrays ``(n_tokens, dim)``, a
            single 2D array (one query), or a 3D array.
        k
            Number of results to return per query.
        """
        self._ensure_loaded()
        self._ensure_mappings()

        self.last_profile = None

        with active().span("query/pack") as _sp:
            queries = self._pack_queries(queries_embeddings)
            if _sp is not None:
                _sp.meta["n_queries"] = int(queries.shape[0])

        # Chimera has no stage instrumentation of its own — the C++ search is
        # opaque from Python. One derived span keeps the E2E breakdown honest
        # about that instead of leaving the whole search in `unaccounted`.
        n_queries = int(queries.shape[0])
        wrapper_start = time.perf_counter_ns()
        if self._scored_search is not None:
            raw = getattr(self._index, self._scored_search)(queries, k=k)
        else:
            raw = self._index.search(queries, k=k)
        wrapper_ns = time.perf_counter_ns() - wrapper_start

        dispatch = Span(
            name="overhead/dispatch",
            count=n_queries,
            dur_ns=wrapper_ns,
            meta={"derived": 1.0, "n_queries": n_queries, "opaque_backend": 1.0},
        )
        self.last_profile = [dispatch]
        active().attach([dispatch])

        # A single 2D query comes back as a flat list; normalise to batch shape.
        if raw and not isinstance(raw[0], (list, tuple)):
            raw = [raw]

        with active().span("result/convert", count=n_queries) as _sp:
            results = []
            for query_results in raw:
                query_docs = []
                for rank, entry in enumerate(query_results):
                    if self._scored_search is not None:
                        score, doc_id = float(entry[0]), int(entry[1])
                    else:
                        # Upstream drops the late-interaction score when it
                        # drains the result heap, but the order it drains in is
                        # descending — so rank is exact and these placeholders
                        # preserve it. See `scores_are_synthetic`.
                        doc_id = int(entry)
                        score = float(len(query_results) - rank)
                    query_docs.append(
                        RerankResult(id=self._int_to_doc_id[doc_id], score=score)
                    )
                results.append(query_docs)
            if _sp is not None:
                _sp.meta["n_results"] = sum(len(r) for r in results)

        return results

    # -- unsupported operations --------------------------------------------

    def remove_documents(self, documents_ids: list[str]) -> None:
        warnings.warn(
            "Chimera does not support document removal. This call has no effect.",
            UserWarning,
            stacklevel=2,
        )

    def get_documents_embeddings(
        self, documents_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        raise NotImplementedError(
            "Chimera stores documents as rotated one-bit and full-bit codes and "
            "exposes no reconstruction path through its bindings."
        )

    def __repr__(self) -> str:
        if not self.is_indexed:
            return f"Chimera(path={self.index_path!r}, empty)"
        return (
            f"Chimera(\n"
            f"  path={self.index_path!r},\n"
            f"  docs={self._params.get('n_documents')}, "
            f"tokens={self._params.get('n_tokens')}, dim={self._params.get('dim')},\n"
            f"  — build —\n"
            f"  n_clusters={self._params.get('n_clusters')}, "
            f"ex_bits={self._params.get('ex_bits')},\n"
            f"  — search —\n"
            f"  nprobe={self.nprobe}, k_refine={self.k_refine}, "
            f"k_full_bit={self.k_full_bit},\n"
            f"  cagra_itopk_size={self.cagra_itopk_size}, "
            f"num_chunks={self.num_chunks},\n"
            f"  scores_are_synthetic={self.scores_are_synthetic}\n"
            f")"
        )
