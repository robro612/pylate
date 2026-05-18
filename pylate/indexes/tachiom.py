from __future__ import annotations

import logging
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from ..rank import RerankResult
from .base import Base

logger = logging.getLogger(__name__)


class Tachiom(Base):
    """Tachiom index using Token-Aware Clustering and hierarchical PQ for multi-vector search.

    Parameters
    ----------
    index_folder
        The folder where the index will be stored.
    index_name
        The name of the index.
    override
        Whether to override the index if it already exists.
    total_centroids
        Total coarse-centroid budget distributed across token types by TAC.
        When ``None`` (default), set to 5 % of the total token count at build
        time. Pass an explicit integer to override.
    tac_n_iter
        Number of k-means iterations inside TAC per token group.
    pq_sample_size
        Number of training vectors for the PQ encoder. Capped at the actual
        number of tokens if the collection is smaller.
    pq_n_iter
        PQ k-means iterations.
    normalize
        L2-normalise residuals before PQ encoding. The per-token norms are
        embedded in the encoded payload so they can be recovered at search time.
    pq_seed
        Random seed for PQ training reproducibility.
    hnsw_m
        HNSW neighbour count (edges per node in the graph).
    ef_construction
        HNSW build-time beam width.
    pq_subspaces
        PQ subspace count. Only 32 is currently supported by tachiom.
    k_centroids
        Coarse centroids retrieved per query token via HNSW during the Gather phase.
    ef_search
        HNSW beam width during search. Increase together with ``k_centroids``
        for deeper search.
    alpha
        After accumulating coarse scores, only documents scoring above
        ``alpha × score_k`` are forwarded to PQ reranking. Lower values prune
        more aggressively. Set to ``None`` to disable.
    k_docs_to_score
        Maximum candidates passed to PQ reranking (cap applied after alpha-pruning).
    beta
        Stop PQ reranking after this many candidates have been scored.
        ``None`` scores all ``k_docs_to_score`` candidates.
    lambda_
        Distance-adaptive HNSW early-exit factor. ``None`` to disable.
    num_threads
        Threads for batch search: 0 = all cores, 1 = serial, n = custom pool.

    Notes
    -----
    Tachiom stores embeddings in compressed/quantized form and does not support
    incremental adds or deletions. Adding documents triggers a full index rebuild
    from all accumulated data. For large corpora, pass all documents in a single
    ``add_documents`` call.

    Passing ``documents_token_ids`` to ``add_documents`` enables Token-Aware
    Clustering, which distributes the centroid budget proportionally across
    vocabulary types. Without token IDs, TAC degrades to standard k-means.
    """

    is_end_to_end_index = True

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "tachiom",
        override: bool = False,
        # Build params
        total_centroids: int | None = None,
        tac_n_iter: int = 10,
        pq_sample_size: int = 10_000_000,
        pq_n_iter: int = 10,
        normalize: bool = False,
        pq_seed: int = 42,
        hnsw_m: int = 32,
        ef_construction: int = 1500,
        pq_subspaces: int = 32,
        # Search params
        k_centroids: int = 20,
        ef_search: int = 30,
        alpha: float | None = 0.45,
        k_docs_to_score: int = 500,
        beta: int | None = None,
        lambda_: float | None = None,
        num_threads: int = 0,
    ) -> None:
        try:
            import tachiom as _tachiom  # noqa: F401
        except ImportError:
            raise ImportError(
                "tachiom is not installed. Please install it with: "
                '`pip install "pylate[tachiom]"` or `pip install tachiom`.'
            )

        self.index_folder = index_folder
        self.index_name = index_name

        # Build hyperparameters
        self.total_centroids = total_centroids
        self.tac_n_iter = tac_n_iter
        self.pq_sample_size = pq_sample_size
        self.pq_n_iter = pq_n_iter
        self.normalize = normalize
        self.pq_seed = pq_seed
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.pq_subspaces = pq_subspaces

        # Search hyperparameters
        self.k_centroids = k_centroids
        self.ef_search = ef_search
        self.alpha = alpha
        self.k_docs_to_score = k_docs_to_score
        self.beta = beta
        self.lambda_ = lambda_
        self.num_threads = num_threads

        # Directory structure
        self.index_path = os.path.join(index_folder, index_name)
        if override and os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path, exist_ok=True)

        self._tachiom_bin = os.path.join(self.index_path, "tachiom_index.bin")
        self._vectors_npy = os.path.join(self.index_path, "vectors.npy")
        self._token_ids_npy = os.path.join(self.index_path, "token_ids.npy")
        self._doclens_npy = os.path.join(self.index_path, "doclens.npy")
        self._doc_ids_pkl = os.path.join(self.index_path, "doc_ids.pkl")

        # doc_ids[i] is the pylate string ID for tachiom's integer document i
        self._doc_ids: list[str] = []
        self._index = None

        self.is_indexed = os.path.exists(self._tachiom_bin) and os.path.exists(
            self._doc_ids_pkl
        )
        if self.is_indexed:
            self._load_doc_ids()
            self._load_index()

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _load_doc_ids(self) -> None:
        with open(self._doc_ids_pkl, "rb") as f:
            self._doc_ids = pickle.load(f)

    def _save_doc_ids(self) -> None:
        with open(self._doc_ids_pkl, "wb") as f:
            pickle.dump(self._doc_ids, f)

    def _load_index(self) -> None:
        import tachiom as _tachiom

        self._index = _tachiom.Tachiom.load(self._tachiom_bin)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents_ids: str | list[str],
        documents_embeddings: list[np.ndarray],
        documents_token_ids: list[np.ndarray] | None = None,
        **kwargs,
    ) -> "Tachiom":
        """Add documents to the index and rebuild it.

        Parameters
        ----------
        documents_ids
            String IDs to associate with the embeddings.
        documents_embeddings
            Per-document token embedding arrays, each of shape ``[n_tokens, dim]``.
            Accepts float32 or float16; float32 is cast to float16.
        documents_token_ids
            Per-document vocabulary ID arrays, each of shape ``[n_tokens]``, dtype
            int64. These are the tokenizer ``input_ids`` for each token after skiplist
            filtering, aligned with the rows of the corresponding embedding array.
            When provided, TAC can cluster each token type separately. When ``None``,
            zeros are used and TAC degrades to ordinary k-means.
        **kwargs
            Accepted for compatibility with the base interface. Ignored.
        """
        import tachiom as _tachiom

        if isinstance(documents_ids, str):
            documents_ids = [documents_ids]

        # Load any previously accumulated raw data
        if os.path.exists(self._vectors_npy):
            prev_vectors = np.load(self._vectors_npy, mmap_mode="r")
            prev_token_ids = np.load(self._token_ids_npy, mmap_mode="r")
            prev_doclens = np.load(self._doclens_npy, mmap_mode="r")
        else:
            dim = documents_embeddings[0].shape[-1]
            prev_vectors = np.empty((0, dim), dtype=np.float16)
            prev_token_ids = np.empty(0, dtype=np.int64)
            prev_doclens = np.empty(0, dtype=np.int32)

        new_vecs, new_tids, new_doclens = [], [], []
        for i, emb in enumerate(documents_embeddings):
            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy()
            emb = np.asarray(emb)
            if emb.dtype != np.float16:
                emb = emb.astype(np.float16)
            emb = np.ascontiguousarray(emb)
            n_tok = emb.shape[0]

            if documents_token_ids is not None:
                tids = np.asarray(documents_token_ids[i], dtype=np.int64)
            else:
                tids = np.zeros(n_tok, dtype=np.int64)

            new_vecs.append(emb)
            new_tids.append(tids)
            new_doclens.append(n_tok)

        all_vectors = np.concatenate(
            [np.asarray(prev_vectors), np.concatenate(new_vecs, axis=0)], axis=0
        ).astype(np.float16)
        all_token_ids = np.concatenate(
            [np.asarray(prev_token_ids), np.concatenate(new_tids, axis=0)]
        ).astype(np.int64)
        all_doclens = np.concatenate(
            [np.asarray(prev_doclens), np.array(new_doclens, dtype=np.int32)]
        ).astype(np.int32)

        # Persist raw arrays (reused if add_documents is called again)
        np.save(self._vectors_npy, np.ascontiguousarray(all_vectors))
        np.save(self._token_ids_npy, np.ascontiguousarray(all_token_ids))
        np.save(self._doclens_npy, np.ascontiguousarray(all_doclens))

        self._doc_ids.extend(documents_ids)
        self._save_doc_ids()

        n_tokens = len(all_token_ids)
        if self.total_centroids is None:
            total_centroids = max(1, int(0.05 * n_tokens))
            logger.info("total_centroids set to 5%% of tokens: %d", total_centroids)
        else:
            total_centroids = min(self.total_centroids, n_tokens)
            if total_centroids < self.total_centroids:
                logger.warning(
                    "total_centroids reduced from %d to %d to match token count",
                    self.total_centroids,
                    total_centroids,
                )

        # Build from the saved files (tachiom reads them with mmap internally)
        logger.info(
            "Building Tachiom index: %d docs, %d tokens, %d centroids",
            len(self._doc_ids),
            n_tokens,
            total_centroids,
        )
        self._index = _tachiom.Tachiom.build(
            self._vectors_npy,
            self._token_ids_npy,
            self._doclens_npy,
            total_centroids=total_centroids,
            tac_n_iter=self.tac_n_iter,
            pq_sample_size=self.pq_sample_size,
            pq_n_iter=self.pq_n_iter,
            normalize=self.normalize,
            pq_seed=self.pq_seed,
            hnsw_m=self.hnsw_m,
            ef_construction=self.ef_construction,
            pq_subspaces=self.pq_subspaces,
        )
        self._index.save(self._tachiom_bin)
        self.is_indexed = True

        return self

    def remove_documents(self, documents_ids: list[str]) -> "Tachiom":
        """Not supported — Tachiom has no incremental deletion.

        Rebuild the index from scratch using ``override=True`` with the desired
        document set.
        """
        raise NotImplementedError(
            "Tachiom does not support incremental document removal. "
            "Rebuild the index with the desired documents using override=True."
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def __call__(
        self,
        queries_embeddings: np.ndarray | list[np.ndarray],
        k: int = 10,
        subset: list[list[str]] | list[str] | None = None,
    ) -> list[list[RerankResult]]:
        """Query the index for the nearest neighbours of the query embeddings.

        Parameters
        ----------
        queries_embeddings
            Either a list of per-query arrays each of shape ``[n_tokens, dim]``
            (variable-length / ragged), or a single stacked array of shape
            ``[n_queries, n_tokens, dim]`` (uniform token count).
        k
            Number of results to return per query.
        subset
            Not supported by Tachiom. Ignored with a warning if provided.

        Returns
        -------
        List of lists of ``RerankResult`` with ``id`` and ``score`` fields.
        """
        if not self.is_indexed:
            raise ValueError(
                "The index is empty. Please add documents before querying."
            )

        if subset is not None:
            logger.warning("Tachiom does not support subset filtering; ignoring.")

        # Normalise input to a list of f32 C-contiguous arrays
        if isinstance(queries_embeddings, np.ndarray) and queries_embeddings.ndim == 3:
            queries_list = list(queries_embeddings)
        elif isinstance(queries_embeddings, list):
            queries_list = queries_embeddings
        else:
            queries_list = [queries_embeddings]

        def _to_f32(q: np.ndarray) -> np.ndarray:
            if hasattr(q, "cpu"):
                q = q.cpu().numpy()
            q = np.asarray(q)
            if q.dtype != np.float32:
                q = q.astype(np.float32)
            return np.ascontiguousarray(q)

        queries_f32 = [_to_f32(q) for q in queries_list]
        n_queries = len(queries_f32)
        sentinel = np.iinfo(np.uint32).max

        # Check whether all queries share the same token count (uniform mode)
        token_counts = [q.shape[0] for q in queries_f32]
        uniform = len(set(token_counts)) == 1

        if uniform:
            # Stack into [total_tokens, dim] for efficient batch_search
            tokens_flat = np.ascontiguousarray(
                np.concatenate(queries_f32, axis=0)
            )
            scores_batch, ids_batch = self._index.batch_search(
                tokens_flat,
                n_queries,
                k=k,
                num_threads=self.num_threads,
                k_centroids=self.k_centroids,
                k_docs_to_score=self.k_docs_to_score,
                ef_search=self.ef_search,
                alpha=self.alpha,
                beta=self.beta,
                lambda_=self.lambda_,
            )
            results = []
            for i in range(n_queries):
                query_docs = []
                for j in range(k):
                    did = int(ids_batch[i, j])
                    if did != sentinel and did < len(self._doc_ids):
                        query_docs.append(
                            RerankResult(
                                id=self._doc_ids[did],
                                score=float(scores_batch[i, j]),
                            )
                        )
                results.append(query_docs)
        else:
            # Ragged mode: build offsets array and concatenate all tokens
            offsets = np.zeros(n_queries + 1, dtype=np.uint64)
            for i, cnt in enumerate(token_counts):
                offsets[i + 1] = offsets[i] + cnt
            tokens_flat = np.ascontiguousarray(np.concatenate(queries_f32, axis=0))
            scores_batch, ids_batch = self._index.batch_search(
                tokens_flat,
                n_queries,
                k=k,
                offsets=offsets,
                num_threads=self.num_threads,
                k_centroids=self.k_centroids,
                k_docs_to_score=self.k_docs_to_score,
                ef_search=self.ef_search,
                alpha=self.alpha,
                beta=self.beta,
                lambda_=self.lambda_,
            )
            results = []
            for i in range(n_queries):
                query_docs = []
                for j in range(k):
                    did = int(ids_batch[i, j])
                    if did != sentinel and did < len(self._doc_ids):
                        query_docs.append(
                            RerankResult(
                                id=self._doc_ids[did],
                                score=float(scores_batch[i, j]),
                            )
                        )
                results.append(query_docs)

        return results

    def get_documents_embeddings(
        self, document_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        """Not supported — Tachiom stores embeddings in compressed/quantized form."""
        raise NotImplementedError(
            "Tachiom does not provide direct access to document embeddings; "
            "they are stored in compressed/quantized form."
        )
