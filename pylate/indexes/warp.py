from __future__ import annotations

import logging
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from ..rank import RerankResult
from .base import Base

logger = logging.getLogger(__name__)


def convert_embeddings_to_torch(
    embeddings: np.ndarray | torch.Tensor | list,
) -> list[torch.Tensor]:
    """Convert embeddings to list of torch tensors as expected by xtr-warp."""
    if isinstance(embeddings, list):
        if len(embeddings) == 0:
            return []
        if isinstance(embeddings[0], torch.Tensor):
            return embeddings
        elif isinstance(embeddings[0], np.ndarray):
            return [torch.from_numpy(emb) for emb in embeddings]

    if isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 3:
            return [torch.from_numpy(embeddings[i]) for i in range(embeddings.shape[0])]
        elif len(embeddings.shape) == 2:
            return [torch.from_numpy(embeddings)]

    if isinstance(embeddings, torch.Tensor):
        if len(embeddings.shape) == 3:
            return [embeddings[i] for i in range(embeddings.shape[0])]
        elif len(embeddings.shape) == 2:
            return [embeddings]

    return embeddings


class WARP(Base):
    """WARP index using the xtr-warp-rs backend for high-performance multi-vector search.

    WARP is a PLAID-like retrieval engine optimized for XTR models, featuring per-token
    centroid pruning, posting-list pruning, and error-aware merging for significantly
    faster search compared to PLAID while maintaining retrieval quality.

    Parameters
    ----------
    index_folder
        The folder where the index will be stored.
    index_name
        The name of the index.
    override
        Whether to override the collection if it already exists.
    nbits
        The number of bits to use for residual quantization (2 or 4).
        Lower values mean more compression but can reduce accuracy.
    kmeans_niters
        The number of iterations for K-means clustering during index creation.
    max_points_per_centroid
        The maximum number of points per centroid during K-means.
    n_samples_kmeans
        The number of samples to use for K-means clustering.
        If None, automatically calculated based on corpus size.
    device
        Device for computation. Can be "cpu", "cuda", "cuda:0", etc.,
        or a list of devices for multi-GPU search.
        If None, defaults to "cuda" if available, else "cpu".
    dtype
        The dtype for loaded index tensors (e.g., torch.float32, torch.float16).
    mmap
        Whether to memory-map large tensors when loading (CPU only).
    use_triton
        Whether to use Triton kernels for K-means. Faster but introduces
        variance due to race conditions. If None, uses Triton if available.
    seed
        Random seed for reproducibility during index creation.
    auto_tune
        If True, automatically tune search hyperparameters after index creation
        using a sample of document embeddings as pseudo-queries. Note: the
        autotuner tends to over-prune (high centroid_score_threshold, low nprobe),
        trading quality for speed. The manual defaults below were chosen via
        parameter sweeps on BEIR datasets to match PLAID/ScaNN quality.
    bound
        Number of centroids to consider per query token during search.
        Default is 256 (8 * nprobe).
    nprobe
        Number of IVF probes per token during search.
        Default is 32.
    t_prime
        Missing token penalty parameter. Controls score compensation
        for tokens that don't match any centroid. Default is 100,000.

    max_candidates
        Maximum number of candidate documents before final sorting.
        Default is 2048.
    centroid_score_threshold
        Per-token centroid filtering threshold in [0, 1]. Lower values
        keep more tokens. 0.0 disables filtering and maximizes quality.
        Default is 0.0.
    batch_size
        Batch size for centroid scoring during search.
    num_threads
        Number of CPU threads for search parallelism.

    Examples
    --------
    >>> from pylate import indexes, models

    >>> index = indexes.WARP(
    ...    index_folder="test_index",
    ...    index_name="warp_xtr",
    ...    override=True,
    ... )

    >>> model = models.ColBERT(
    ...    model_name_or_path="lightonai/GTE-ModernColBERT-v1",
    ... )

    >>> documents_embeddings = model.encode([
    ...    "Document content here...",
    ...    "Another document...",
    ... ] * 10, is_query=False)

    >>> index = index.add_documents(
    ...    documents_ids=range(len(documents_embeddings)),
    ...    documents_embeddings=documents_embeddings,
    ... )

    >>> queries_embeddings = model.encode(
    ...     ["search query", "hello world"],
    ...     is_query=True,
    ... )

    >>> scores = index(
    ...     queries_embeddings,
    ...     k=10,
    ... )
    """

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "warp",
        override: bool = False,
        nbits: int = 4,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        n_samples_kmeans: int | None = None,
        device: str | list[str] | None = None,
        dtype: torch.dtype = torch.float32,
        mmap: bool = True,
        use_triton: bool | None = None,
        seed: int = 42,
        auto_tune: bool = False,
        bound: int = 256,
        nprobe: int = 32,
        t_prime: int = 100000,
        max_candidates: int = 2048,
        centroid_score_threshold: float = 0.0,
        batch_size: int | None = 8192,
        num_threads: int | None = 1,
    ) -> None:
        self.index_folder = index_folder
        self.index_name = index_name
        self.nbits = nbits
        self.kmeans_niters = kmeans_niters
        self.max_points_per_centroid = max_points_per_centroid
        self.n_samples_kmeans = n_samples_kmeans
        self.dtype = dtype
        self.mmap = mmap
        self.use_triton = use_triton
        self.seed = seed
        self.auto_tune = auto_tune
        self.batch_size = batch_size
        self.num_threads = num_threads

        # Search hyperparameters (None = auto-tuned or library default)
        self.bound = bound
        self.nprobe = nprobe
        self.t_prime = t_prime
        self.max_candidates = max_candidates
        self.centroid_score_threshold = centroid_score_threshold

        # Resolve device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Create the index directory structure
        self.index_path = os.path.join(index_folder, index_name)
        self.warp_index_path = os.path.join(self.index_path, "warp_index")
        if override:
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)

        os.makedirs(self.index_path, exist_ok=True)

        # Pickle mappings for document IDs
        self.documents_ids_to_warp_ids_path = os.path.join(
            self.index_path, "documents_ids_to_warp_ids.pkl"
        )
        self.warp_ids_to_documents_ids_path = os.path.join(
            self.index_path, "warp_ids_to_documents_ids.pkl"
        )

        # Initialize the XTRWarp index
        from xtr_warp import XTRWarp

        self._warp = XTRWarp(index=self.warp_index_path)
        self._tuned = False

        # Check if index already exists
        self.is_indexed = os.path.exists(self.documents_ids_to_warp_ids_path)

        # If index exists, load it
        if self.is_indexed:
            load_device = self.device if isinstance(self.device, (str, list)) else "auto"
            self._warp.load(device=load_device, dtype=self.dtype, mmap=self.mmap)

    def _load_documents_ids_to_warp_ids(self) -> dict:
        """Load the pickle file that maps document IDs to WARP IDs."""
        if os.path.exists(self.documents_ids_to_warp_ids_path):
            with open(self.documents_ids_to_warp_ids_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _load_warp_ids_to_documents_ids(self) -> dict:
        """Load the pickle file that maps WARP IDs to document IDs."""
        if os.path.exists(self.warp_ids_to_documents_ids_path):
            with open(self.warp_ids_to_documents_ids_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _auto_tune(
        self,
        queries_embeddings: list[torch.Tensor],
        k: int,
    ) -> None:
        """Auto-tune search hyperparameters using real query embeddings."""
        logger.info("Auto-tuning WARP search hyperparameters.")
        # Pad a sample of queries to uniform length for optimize_hyperparams
        n_tune = min(100, len(queries_embeddings))
        sample = queries_embeddings[:n_tune]
        max_len = max(t.shape[0] for t in sample)
        dim = sample[0].shape[1]
        tune_batch = torch.zeros(n_tune, max_len, dim)
        for i, t in enumerate(sample):
            tune_batch[i, : t.shape[0]] = t

        tuned = self._warp.optimize_hyperparams(
            top_k=k, queries_embeddings=tune_batch
        )
        if tuned is not None:
            if self.bound is None:
                self.bound = tuned[0]
            if self.nprobe is None:
                self.nprobe = tuned[1]
            if self.centroid_score_threshold is None:
                self.centroid_score_threshold = tuned[2]
            if self.max_candidates is None:
                self.max_candidates = tuned[3]
            if self.t_prime is None:
                self.t_prime = tuned[4]
            logger.info(
                "Tuned hyperparameters: bound=%s, nprobe=%s, "
                "centroid_score_threshold=%s, max_candidates=%s, t_prime=%s",
                self.bound,
                self.nprobe,
                self.centroid_score_threshold,
                self.max_candidates,
                self.t_prime,
            )
        self._tuned = True

    def _save_mappings(
        self,
        documents_ids_to_warp_ids: dict,
        warp_ids_to_documents_ids: dict,
    ) -> None:
        """Save the ID mappings to pickle files."""
        with open(self.documents_ids_to_warp_ids_path, "wb") as f:
            pickle.dump(documents_ids_to_warp_ids, f)
        with open(self.warp_ids_to_documents_ids_path, "wb") as f:
            pickle.dump(warp_ids_to_documents_ids, f)

    @staticmethod
    def _count_documents_from_doclens(embeddings_path: Path) -> int:
        """Count total documents from .doclens.npy sidecar files."""
        if embeddings_path.is_file():
            doclens_path = embeddings_path.with_suffix(".doclens.npy")
            return len(np.load(doclens_path))
        doclens_files = sorted(embeddings_path.glob("*.doclens.npy"))
        if not doclens_files:
            raise FileNotFoundError(
                f"No .doclens.npy files found in {embeddings_path}"
            )
        return sum(len(np.load(f)) for f in doclens_files)

    def add_documents(
        self,
        documents_ids: str | list[str],
        documents_embeddings: list[np.ndarray | torch.Tensor] | str | Path,
        **kwargs,
    ) -> "WARP":
        """Add documents to the index.

        Note: WARP does not support incremental updates. Calling this method
        on an already-indexed collection will raise an error. To re-index,
        create a new WARP instance with ``override=True``.

        Parameters
        ----------
        documents_ids
            The document IDs corresponding to each embedding.
        documents_embeddings
            List of document embeddings, each with shape ``(num_tokens, embedding_dim)``.
            Alternatively, a path to a directory of ``.npy`` + ``.doclens.npy``
            shard files. When a path is given, embeddings are streamed from disk
            instead of loaded into memory.
        """
        if self.is_indexed:
            raise ValueError(
                "WARP does not support incremental indexing. "
                "Create a new WARP index with override=True to re-index."
            )

        if isinstance(documents_ids, str):
            documents_ids = [documents_ids]

        # Determine whether to use disk path or in-memory embeddings
        use_disk = isinstance(documents_embeddings, (str, Path))
        if use_disk:
            embeddings_path = Path(documents_embeddings)
            embeddings_source = embeddings_path
            num_documents = self._count_documents_from_doclens(embeddings_path)
        else:
            embeddings_source = convert_embeddings_to_torch(documents_embeddings)
            num_documents = len(embeddings_source)

        # Resolve device for creation (must be a single string)
        create_device = self.device
        if isinstance(create_device, list):
            create_device = create_device[0]

        logger.info(
            "Creating WARP index (%s, %d documents).",
            "disk" if use_disk else "memory",
            num_documents,
        )
        self._warp.create(
            embeddings_source=embeddings_source,
            device=create_device,
            kmeans_niters=self.kmeans_niters,
            max_points_per_centroid=self.max_points_per_centroid,
            nbits=self.nbits,
            n_samples_kmeans=self.n_samples_kmeans,
            seed=self.seed,
            use_triton_kmeans=self.use_triton,
        )

        # Load the index for searching
        load_device = self.device if isinstance(self.device, (str, list)) else "auto"
        self._warp.load(device=load_device, dtype=self.dtype, mmap=self.mmap)

        self._tuned = False

        # Store ID mappings
        warp_ids = list(range(num_documents))
        documents_ids_to_warp_ids = dict(zip(documents_ids, warp_ids))
        warp_ids_to_documents_ids = dict(zip(warp_ids, documents_ids))
        self._save_mappings(documents_ids_to_warp_ids, warp_ids_to_documents_ids)

        self.is_indexed = True
        return self

    def remove_documents(self, documents_ids: list[str]) -> "WARP":
        """Remove documents from the index.

        Note: WARP does not support document removal. This method raises
        NotImplementedError. To remove documents, rebuild the index without them.
        """
        raise NotImplementedError(
            "WARP does not support document removal. "
            "Rebuild the index without the unwanted documents using override=True."
        )

    def __call__(
        self,
        queries_embeddings: np.ndarray
        | torch.Tensor
        | list[np.ndarray]
        | list[torch.Tensor],
        k: int = 10,
    ) -> list[list[RerankResult]]:
        """Query the index for the nearest neighbors of the query embeddings.

        Parameters
        ----------
        queries_embeddings
            The query embeddings. Can be a numpy array, torch tensor,
            or list of numpy arrays/torch tensors.
        k
            The number of nearest neighbors to return.

        Returns
        -------
        List of lists containing dictionaries with 'id' and 'score' keys.
        """
        if not self.is_indexed:
            raise ValueError(
                "The index is empty. Please add documents before querying."
            )

        warp_ids_to_documents_ids = self._load_warp_ids_to_documents_ids()

        queries_embeddings = convert_embeddings_to_torch(queries_embeddings)

        # Lazy auto-tune on first search using real queries
        if self.auto_tune and not self._tuned:
            self._auto_tune(queries_embeddings, k)

        # Stack into batch tensor for xtr-warp search
        # xtr-warp accepts list[torch.Tensor] or a single batched tensor
        search_results = self._warp.search(
            queries_embeddings=queries_embeddings,
            top_k=k,
            num_threads=self.num_threads,
            bound=self.bound,
            nprobe=self.nprobe,
            t_prime=self.t_prime,
            max_candidates=self.max_candidates,
            centroid_score_threshold=self.centroid_score_threshold,
            batch_size=self.batch_size,
        )

        # Convert results to RerankResult format
        results = []
        for query_results in search_results:
            query_docs = []
            for warp_id, score in query_results:
                if warp_id in warp_ids_to_documents_ids:
                    doc_id = warp_ids_to_documents_ids[warp_id]
                    query_docs.append(RerankResult(id=doc_id, score=float(score)))
            results.append(query_docs)

        return results

    def get_documents_embeddings(
        self, document_ids: list[list[str]]
    ) -> list[list[list[int | float]]]:
        """Get document embeddings by their IDs.

        Note: WARP stores embeddings in compressed/quantized form.
        This method is not supported.
        """
        raise NotImplementedError(
            "WARP does not provide direct access to document embeddings. "
            "The embeddings are stored in compressed/quantized form."
        )

    def free(self) -> None:
        """Unload the index from memory/GPU."""
        self._warp.free()
