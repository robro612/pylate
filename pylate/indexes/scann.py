from __future__ import annotations

import itertools
import logging
import time

import numpy as np
import torch

from .base import Base

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



    """

    def __init__(
        self,
        name: str | None = "ScaNN_index",
        embedding_size: int = 128,
        num_neighbors: int = 10,
        num_leaves: int | None = None,
        num_leaves_to_search: int | None = None,
        training_sample_size: int = 10000,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.embedding_size = embedding_size
        self.num_neighbors = num_neighbors
        self.verbose = verbose
        
        # Auto-tune parameters based on dataset size (will be set when we know the size)
        # Defaults are conservative for small datasets
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        self.training_sample_size = training_sample_size

        # In-memory data structures only (no file I/O)
        self.searcher = None
        self.all_embeddings = None
        self.embedding_id_to_position = {}  # Map embedding ID to position in array
        self.position_to_embedding_id = {}  # Reverse mapping: position to embedding ID
        self.embeddings_to_documents_ids = {}  # In-memory mapping: embedding ID -> document ID
        self.documents_ids_to_embeddings = {}  # In-memory mapping: document ID -> list of embedding IDs
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

        # Assume embeddings are already normalized
        step_start = time.time()
        embeddings = embeddings.astype(np.float32)
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Converting embeddings to float32 ({embeddings.shape[0]} vectors, {embeddings.shape[1]} dims): {step_time:.4f}s")

        # Auto-tune parameters if not set
        num_vectors = embeddings.shape[0]
        if self.num_leaves is None:
            # For small datasets, use fewer leaves for faster search
            # For large datasets, use more leaves for better accuracy
            if num_vectors < 100_000:
                self.num_leaves = min(16, num_vectors // 1000)
            elif num_vectors < 1_000_000:
                self.num_leaves = min(64, num_vectors // 10000)
            else:
                self.num_leaves = 100
            if self.verbose:
                logger.info(f"[ScaNN] Auto-tuned num_leaves={self.num_leaves} for {num_vectors} vectors")
        
        if self.num_leaves_to_search is None:
            # Search more leaves for better recall, but balance with speed
            if num_vectors < 100_000:
                self.num_leaves_to_search = min(4, self.num_leaves)
            elif num_vectors < 1_000_000:
                self.num_leaves_to_search = min(10, self.num_leaves // 2)
            else:
                self.num_leaves_to_search = min(20, self.num_leaves // 5)
            if self.verbose:
                logger.info(f"[ScaNN] Auto-tuned num_leaves_to_search={self.num_leaves_to_search}")

        # Build ScaNN searcher
        step_start = time.time()
        if self.verbose:
            logger.info(f"[ScaNN] Building ScaNN searcher with {embeddings.shape[0]} vectors...")
            logger.info(f"[ScaNN]   Parameters: num_leaves={self.num_leaves}, num_leaves_to_search={self.num_leaves_to_search}, num_neighbors={self.num_neighbors}")
        searcher = (
            scann.scann_ops_pybind.builder(embeddings, self.num_neighbors, "dot_product")
            .tree(num_leaves=self.num_leaves, num_leaves_to_search=self.num_leaves_to_search, training_sample_size=self.training_sample_size)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(self.num_neighbors)
            .build()
        )
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] ScaNN searcher built: {step_time:.4f}s")

        self.searcher = searcher
        self.all_embeddings = embeddings
        
        total_time = time.time() - build_start
        if self.verbose:
            logger.info(f"[ScaNN] Total searcher build time: {total_time:.4f}s")

    def add_documents(
        self,
        documents_ids: list[str],
        documents_embeddings: list[list[list[int | float]]],
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
        
        step_start = time.time()
        documents_embeddings = reshape_embeddings(embeddings=documents_embeddings)
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Reshaping document embeddings: {step_time:.4f}s")

        # Flatten all document embeddings at once (no need to batch since we rebuild the index)
        step_start = time.time()
        flattened_embeddings = list(itertools.chain(*documents_embeddings))
        
        # Convert to numpy array
        if isinstance(flattened_embeddings[0], list):
            flattened_embeddings = np.array(flattened_embeddings, dtype=np.float32)
        else:
            flattened_embeddings = np.array(flattened_embeddings, dtype=np.float32)
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Flattening and converting to numpy array ({len(flattened_embeddings)} embeddings): {step_time:.4f}s")

        # Assign embedding IDs sequentially starting from 0
        step_start = time.time()
        embedding_ids = list(range(len(flattened_embeddings)))
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Assigning embedding IDs: {step_time:.4f}s")

        # Store mappings in memory
        step_start = time.time()
        total = 0
        for doc_id, document_embeddings in zip(documents_ids, documents_embeddings):
            num_tokens = len(document_embeddings)
            document_embeddings_ids = embedding_ids[total : total + num_tokens]
            self.documents_ids_to_embeddings[doc_id] = document_embeddings_ids

            # Update in-memory mapping
            for emb_id in document_embeddings_ids:
                self.embeddings_to_documents_ids[str(emb_id)] = doc_id
            
            total += num_tokens

        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Storing ID mappings: {step_time:.4f}s")

        # Build the ScaNN index with all embeddings
        if len(flattened_embeddings) > 0:
            if self.verbose:
                logger.info(f"[ScaNN] Building index with {len(flattened_embeddings)} embeddings...")
            
            # Set position mappings sequentially (embedding ID == position for simplicity)
            for pos, emb_id in enumerate(embedding_ids):
                self.embedding_id_to_position[emb_id] = pos
                self.position_to_embedding_id[pos] = emb_id

            # Build searcher (in-memory only)
            self._build_searcher(flattened_embeddings)
            
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
        
        # Use in-memory mapping (already loaded, no need to load from SQLite)
        embeddings_to_documents_ids = self.embeddings_to_documents_ids
        
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
            list(itertools.chain(*queries_embeddings)), dtype=np.float32
        )
        n_tokens_total = len(flattened_queries)
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Flattening {n_tokens_total} query tokens: {step_time:.4f}s")

        # Query the index
        step_start = time.time()
        neighbors, distances = self.searcher.search_batched(flattened_queries, final_num_neighbors=k)
        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] ScaNN search_batched for {n_tokens_total} tokens (k={k}): {step_time:.4f}s ({step_time/n_tokens_total*1000:.2f}ms per token)")

        # Map embedding indices back to document IDs
        step_start = time.time()
        n_tokens_per_query = [len(q) for q in queries_embeddings]
        
        documents = []
        distances_list = []
        
        mapping_time = 0
        lookup_time = 0
        validation_time = 0
        
        query_idx = 0
        for query_num, n_tokens in enumerate(n_tokens_per_query):
            query_documents = []
            query_distances = []
            
            for token_idx in range(n_tokens):
                token_start = time.time()
                token_neighbors = neighbors[query_idx]
                token_distances = distances[query_idx]
                mapping_time += time.time() - token_start
                
                # Map embedding indices to document IDs
                token_docs = []
                token_dists = []
                
                for neighbor_pos, dist in zip(token_neighbors, token_distances):
                    lookup_start = time.time()
                    # neighbor_pos is a position in the array, map back to embedding ID using reverse mapping
                    emb_id = self.position_to_embedding_id.get(neighbor_pos)
                    lookup_time += time.time() - lookup_start
                    
                    validation_start = time.time()
                    if emb_id is not None:
                        # Direct lookup in in-memory dict (no validation needed - if it's in the dict, it's valid)
                        emb_id_str = str(emb_id)
                        doc_id = embeddings_to_documents_ids.get(emb_id_str)
                        if doc_id is not None:
                            token_docs.append(doc_id)
                            token_dists.append(float(dist))
                    validation_time += time.time() - validation_start
                
                query_documents.append(token_docs[:k])
                query_distances.append(token_dists[:k])
                query_idx += 1
            
            documents.append(query_documents)
            distances_list.append(query_distances)

        step_time = time.time() - step_start
        if self.verbose:
            logger.info(f"[ScaNN] Mapping results to document IDs: {step_time:.4f}s")
            logger.info(f"[ScaNN]   - Array indexing: {mapping_time:.4f}s")
            logger.info(f"[ScaNN]   - Position->ID lookup: {lookup_time:.4f}s")
            logger.info(f"[ScaNN]   - Validation & doc ID lookup: {validation_time:.4f}s")
        
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
        """Retrieve document embeddings for re-ranking from ScaNN.
        
        Returns list of lists of numpy arrays, where each array has shape (seq_len, dim).
        """

        if self.all_embeddings is None:
            raise ValueError("Index is empty, add documents before retrieving embeddings.")

        # Retrieve embeddings from memory using their IDs
        reconstructed_embeddings = []
        for doc_group in documents_ids:
            group_embeddings = []
            for doc_id in doc_group:
                doc_embedding_ids = self.documents_ids_to_embeddings[doc_id]
                # Directly index into the array - no conversion needed
                positions = [self.embedding_id_to_position[emb_id] for emb_id in doc_embedding_ids]
                doc_embeddings = self.all_embeddings[positions]  # Shape: (seq_len, dim)
                group_embeddings.append(doc_embeddings)
            reconstructed_embeddings.append(group_embeddings)

        return reconstructed_embeddings

