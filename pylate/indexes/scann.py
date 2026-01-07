from __future__ import annotations

import itertools
import logging
import time
from typing import Optional
import numpy as np
import torch
from tqdm.auto import tqdm

from .base import Base
from .utils import log_memory

logger = logging.getLogger(__name__)


def reshape_embeddings(
    embeddings: np.ndarray | torch.Tensor | list,
) -> np.ndarray | list:
    """Reshape the embeddings, the ScaNN index expects arrays with shape batch_size, n_tokens, embedding_size."""
    if isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 2:
            return np.expand_dims(a=embeddings, axis=0)

    if isinstance(embeddings, torch.Tensor):
        return reshape_embeddings(embeddings=embeddings.cpu().detach().numpy())

    if isinstance(embeddings, list) and isinstance(embeddings[0], torch.Tensor):
        return [embedding.cpu().detach().numpy() for embedding in embeddings]

    return embeddings


class ScaNN(Base):
    """ScaNN index. The ScaNN index is a fast and efficient index for approximate nearest neighbor search.

    **Important Notes:**
    - ScaNN is an **approximate** nearest neighbor search (not exact), designed for large-scale datasets
    - For ColBERT retrieval, PLAID is typically faster and more accurate as it's optimized for ColBERT scoring
    - ScaNN is CPU-only (no GPU acceleration)
    - Parameters are auto-tuned based on dataset size if not specified

    To use this index, you need to install the `scann` extra:

    ```bash
    pip install "pylate[scann]"
    ```

    or install scann directly:

    ```bash
    pip install scann
    ```

    Parameters
    ----------
    name
        The name of the index collection.
    embedding_size
        The number of dimensions of the embeddings.
    num_neighbors
        The number of neighbors to use for the ScaNN searcher.
    num_leaves
        The number of leaves in the ScaNN tree. If None, auto-tuned based on dataset size.
        For small datasets (<100K vectors), fewer leaves are used for speed.
    num_leaves_to_search
        The number of leaves to search during query time. If None, auto-tuned based on dataset size.
        Higher values improve recall but slow down search.
    training_sample_size
        The number of samples to use for training the ScaNN index.
    verbose
        Whether to enable verbose logging of timing and operations.
        Defaults to False for cleaner output.
    use_autopilot
        Whether to use ScaNN's autopilot() method for automatic parameter tuning.
        If True, overrides num_leaves, num_leaves_to_search, and training_sample_size.
        Defaults to False.



    """

    def __init__(
        self,
        name: str | None = "ScaNN_index",
        embedding_size: int = 128,
        num_neighbors: Optional[int] = 10,
        num_leaves: Optional[int] = None,
        num_leaves_to_search: Optional[int] = None,
        training_sample_size: Optional[int] = None,
        verbose: bool = False,
        use_autopilot: bool = False,
        store_embeddings: bool = False,
    ) -> None:
        self.name = name
        self.embedding_size = embedding_size
        self.num_neighbors = num_neighbors
        self.verbose = verbose
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        self.training_sample_size = training_sample_size
        self.use_autopilot = use_autopilot
        self.store_embeddings = store_embeddings
        # In-memory data structures only (no file I/O)
        self.searcher = None
        # Note: embedding_id == position (sequential IDs), so no need for separate mappings
        # Store (start, length) tuples instead of lists for memory efficiency
        self.doc_id_to_embedding_range = {}  # doc_id -> (start_position, length) tuple
        self.position_to_doc_id = None  # Direct mapping: position -> document ID (numpy array for vectorized indexing)
        self.flattened_embeddings = None  # Flattened embeddings array (only if store_embeddings=True)
        self._documents_added = False  # Track if documents have been added

    def _build_searcher(self, embeddings: np.ndarray) -> None:
        """Build the ScaNN searcher from embeddings (in-memory only)."""
        build_start = time.time()
        try:
            import scann
        except ImportError:
            raise ImportError(
                'ScaNN is not installed. Please install it with: `pip install "pylate[scann]"` or `pip install scann`.'
            )

        # Auto-tune parameters if not set (only if not using autopilot)
        num_vectors = embeddings.shape[0]
        self.num_neighbors = self.num_neighbors if self.num_neighbors else min(10, num_vectors)
        
        if self.use_autopilot:
            # When using autopilot, it will auto-tune all parameters
            if self.verbose:
                logger.info(f"[ScaNN] Building ScaNN searcher with {embeddings.shape[0]} vectors using autopilot()...")
                logger.info(f"[ScaNN]   NOTE: autopilot() overrides manual configuration (num_leaves, num_leaves_to_search, training_sample_size)")
                if self.num_leaves is not None or self.num_leaves_to_search is not None or self.training_sample_size is not None:
                    logger.warning(f"[ScaNN]   WARNING: Manual parameters provided but will be ignored: num_leaves={self.num_leaves}, num_leaves_to_search={self.num_leaves_to_search}, training_sample_size={self.training_sample_size}")
        else:
            # Auto-tune parameters if not set
            self.num_leaves = self.num_leaves if self.num_leaves else min(2_000, num_vectors)
            self.num_leaves_to_search = self.num_leaves_to_search if self.num_leaves_to_search else 200
            self.training_sample_size = self.training_sample_size if self.training_sample_size else min(250000, num_vectors)

            if self.verbose:
                logger.info(f"[ScaNN] Building ScaNN searcher with {embeddings.shape[0]} vectors...")
                logger.info(f"[ScaNN]   Parameters: num_leaves={self.num_leaves}, num_leaves_to_search={self.num_leaves_to_search}, training_sample_size={self.training_sample_size}, num_neighbors={self.num_neighbors}")

        # Build ScaNN searcher
        log_memory("Before scann.build()", self.verbose)
        step_start = time.time()
        if self.use_autopilot:
            searcher = (
                scann.scann_ops_pybind.builder(embeddings, self.num_neighbors, "dot_product")
                .autopilot()
                .build()
            )
        else:
            searcher = (
                scann.scann_ops_pybind.builder(embeddings, self.num_neighbors, "dot_product")
                .tree(num_leaves=self.num_leaves, num_leaves_to_search=self.num_leaves_to_search, training_sample_size=self.training_sample_size, spherical=True)
                .score_ah(1, anisotropic_quantization_threshold=0.1)
                .build()
            )
        step_time = time.time() - step_start
        log_memory("After scann.build()", self.verbose)
        if self.verbose:
            logger.info(f"[ScaNN] ScaNN searcher built: {step_time:.4f}s")

        self.searcher = searcher
        self.index_config = searcher.config()
        
        total_time = time.time() - build_start
        if self.verbose:
            logger.info(f"[ScaNN] Total searcher build time: {total_time:.4f}s")

    def add_documents(
        self,
        documents_ids: list[str],
        documents_embeddings: list[torch.Tensor],
        batch_size: int,
    ) -> None:
        """Add documents to the index.
        
        Note: This method only supports adding all documents at once. 
        Subsequent calls will raise an error.
        batch_size is kept for API compatibility but not used.
        """
        # Enforce single add - check if documents already exist
        if self._documents_added:
            raise ValueError(
                "ScaNN index only supports adding all documents at once. "
                "Documents have already been added."
            )
        
        add_start = time.time()
        if self.verbose:
            logger.info(f"[ScaNN] Adding {len(documents_ids)} documents to index...")
        
        log_memory("Start of add_documents", self.verbose)

        # Calculate total embeddings to pre-allocate array
        # Assumes input is list of torch tensors (the standard pylate format)
        step_start = time.time()
        import gc
        
        # Get doc lengths and total count in one pass
        doc_lengths = [emb.shape[0] for emb in documents_embeddings]
        total_embeddings = sum(doc_lengths)
        embedding_dim = documents_embeddings[0].shape[1]
        
        if self.verbose:
            logger.info(f"[ScaNN] Pre-allocating array for {total_embeddings} embeddings x {embedding_dim} dims ({total_embeddings * embedding_dim * 4 / 1e9:.2f} GB)")
        
        # Pre-allocate final float32 array (ScaNN requires float32)
        flattened_embeddings = np.empty((total_embeddings, embedding_dim), dtype=np.float32)
        
        log_memory("After pre-allocating flattened_embeddings array", self.verbose)
        
        # Fill array in-place, deleting each tensor after copying to free memory
        offset = 0
        num_docs = len(documents_embeddings)
        log_interval = max(1, num_docs // 10)  # Log memory ~10 times during the loop
        
        iterator = tqdm(
            enumerate(documents_embeddings),
            desc="Flattening documents and adding to pre-allocated array",
            total=num_docs,
            disable=not self.verbose,
        )
        for i, emb in iterator:
            n = emb.shape[0]
            flattened_embeddings[offset:offset + n] = emb.to("cpu", dtype=torch.float32).numpy()
            offset += n
            
            # Log memory periodically
            if self.verbose and (i + 1) % log_interval == 0:
                log_memory(f"During fill loop ({i + 1}/{num_docs} docs, {offset}/{total_embeddings} embeddings)", self.verbose)
        
        log_memory("After fill loop, before gc", self.verbose)
        
        # Clear the list and run gc
        del documents_embeddings
        gc.collect()
        
        log_memory("After del documents_embeddings + gc.collect()", self.verbose)
        
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Flattened {total_embeddings} embeddings to float32: {step_time:.4f}s")
        
        # Build position->doc_id array and doc_id->embedding_range mapping
        step_start = time.time()
        self.position_to_doc_id = np.empty(total_embeddings, dtype=object)
        offset = 0
        for doc_id, num_tokens in zip(documents_ids, doc_lengths):
            # Store (start, length) tuple instead of list for memory efficiency
            self.doc_id_to_embedding_range[doc_id] = (offset, num_tokens)
            # Broadcast doc_id to fill the slice (no temp list needed)
            self.position_to_doc_id[offset:offset + num_tokens] = doc_id
            offset += num_tokens
        
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Built ID mappings and position->doc_id array: {step_time:.4f}s")

        # Build the ScaNN index with all embeddings
        if len(flattened_embeddings) > 0:
            if self.verbose:
                logger.info(f"[ScaNN] Building index with {len(flattened_embeddings)} embeddings...")
            
            log_memory("Before _build_searcher", self.verbose)
            
            # Note: embedding_id == position (sequential), so no position mappings needed
            # Build searcher (in-memory only)
            self._build_searcher(flattened_embeddings)
            
            # Store flattened embeddings if requested, otherwise free the array
            if self.store_embeddings:
                self.flattened_embeddings = flattened_embeddings
                log_memory("After _build_searcher + storing flattened_embeddings reference", self.verbose)
            else:
                del flattened_embeddings
                gc.collect()
                log_memory("After _build_searcher + del flattened_embeddings", self.verbose)
            
            # Mark that documents have been added
            self._documents_added = True
            
            total_time = time.time() - add_start
            if self.verbose:
                logger.info(f"[ScaNN] Total add_documents time: {total_time:.4f}s")

    def remove_documents(self, documents_ids: list[str]) -> None:
        """Remove documents from the index.
        
        Not supported for ScaNN index.

        Parameters
        ----------
        documents_ids
            The documents IDs to remove.

        Raises
        ------
        NotImplementedError
            Document removal is not supported for ScaNN index.

        """
        raise NotImplementedError(
            "Document removal is not supported for ScaNN index."
        )

    def __call__(
        self,
        queries_embeddings: list[list[int | float]],
        k: int = 5,
        subset: list[list[str]] | list[str] | None = None,
    ) -> dict:
        """Query the index for the nearest neighbors of the queries embeddings.

        Parameters
        ----------
        queries_embeddings
            The queries embeddings.
        k
            The number of nearest neighbors to return.
        subset
            Optional subset of document IDs to restrict search to.
            Not yet implemented for ScaNN index.

        Raises
        ------
        NotImplementedError
            If subset is provided (not yet implemented).

        """
        if subset is not None:
            raise NotImplementedError(
                "Subset filtering is not yet implemented for ScaNN index."
            )
        
        if self.searcher is None:
            raise ValueError("Index is empty, add documents before querying.")

        total_start = time.time()
        
        # Reshape queries
        step_start = time.time()
        queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
        n_queries = len(queries_embeddings)
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Reshaping {n_queries} queries: {step_time:.4f}s")

        # Flatten query embeddings (assume they are already normalized)
        step_start = time.time()
        flattened_queries = np.array(
            list(itertools.chain(*queries_embeddings))
        )
        n_tokens_total = len(flattened_queries)
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Flattening {n_tokens_total} query tokens: {step_time:.4f}s")

        # Query the index
        step_start = time.time()
        neighbors, distances = self.searcher.search_batched_parallel(flattened_queries, final_num_neighbors=k)
        # replace NaN values with 0
        if np.isnan(distances).any():
            print(f"distances has {np.isnan(distances).sum()} NaN values out of {distances.size} total values")
            distances = np.nan_to_num(distances, nan=0.0)
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] ScaNN search_batched for {n_tokens_total} tokens (k={k}): {step_time:.4f}s ({step_time/n_tokens_total*1000:.2f}ms per token)")

        # Map embedding indices back to document IDs using fully vectorized numpy operations
        step_start = time.time()
        n_tokens_per_query = [len(q) for q in queries_embeddings]
        
        # Vectorized lookup: process all tokens at once using numpy advanced indexing
        # neighbors shape: (n_tokens_total, k), distances shape: (n_tokens_total, k)
        all_neighbor_positions = neighbors[:, :k]  # Shape: (n_tokens_total, k)
        all_doc_ids = self.position_to_doc_id[all_neighbor_positions]  # Vectorized lookup for all tokens
        all_distances = distances[:, :k]  # Vectorized conversion
        
        # Reshape back into nested structure (queries -> tokens -> neighbors)
        documents = []
        distances_list = []
        token_idx = 0
        for query_num, n_tokens in enumerate(n_tokens_per_query):
            query_documents = []
            query_distances = []
            
            for _ in range(n_tokens):
                # Extract results for this token (already vectorized)
                token_docs = all_doc_ids[token_idx]
                token_dists = all_distances[token_idx]
                
                query_documents.append(token_docs)
                query_distances.append(token_dists)
                token_idx += 1
            
            documents.append(query_documents)
            distances_list.append(query_distances)

        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Mapping results to document IDs: {step_time:.4f}s")
        
        total_time = time.time() - total_start
        if self.verbose:
            logger.info(f"[ScaNN] Total retrieval time: {total_time:.4f}s ({total_time/n_queries*1000:.2f}ms per query, {total_time/n_tokens_total*1000:.2f}ms per token)")

        return {
            "documents_ids": documents,
            "distances": np.array(distances_list),
        }

    def get_documents_embeddings(
        self, documents_ids: list[list[str]]
    ) -> list[list[np.ndarray]]:
        """Get document embeddings by their IDs.
        
        Parameters
        ----------
        documents_ids
            Nested list of document IDs. Each inner list represents a group of documents.
        
        Returns
        -------
        list[list[np.ndarray]]
            Nested list of embeddings. Each embedding is a numpy array with shape (seq_len, dim).
        
        Raises
        ------
        NotImplementedError
            If store_embeddings=False (embeddings are not stored).
        ValueError
            If index is empty or document ID not found.
        """
        if not self.store_embeddings:
            raise NotImplementedError(
                "Retrieving document embeddings requires store_embeddings=True. "
                "Set store_embeddings=True when creating the index."
            )
        
        if self.flattened_embeddings is None:
            raise ValueError("Index is empty, add documents before retrieving embeddings.")
        
        reconstructed_embeddings = []
        for doc_group in documents_ids:
            group_embeddings = []
            for doc_id in doc_group:
                if doc_id not in self.doc_id_to_embedding_range:
                    raise ValueError(f"Document ID '{doc_id}' not found in index.")
                
                start, length = self.doc_id_to_embedding_range[doc_id]
                # Slice the flattened array to get document embeddings
                doc_emb = self.flattened_embeddings[start:start + length]
                group_embeddings.append(doc_emb)
            reconstructed_embeddings.append(group_embeddings)
        
        return reconstructed_embeddings
