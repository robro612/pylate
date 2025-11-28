from __future__ import annotations

import itertools
import logging

import numpy as np
import torch

from .base import Base

logger = logging.getLogger(__name__)


def reshape_embeddings(
    embeddings: np.ndarray | torch.Tensor | list,
    device: str | torch.device | None = None,
) -> torch.Tensor | list:
    """Reshape the embeddings, expects arrays with shape batch_size, n_tokens, embedding_size."""
    if isinstance(embeddings, np.ndarray):
        tensor = torch.from_numpy(embeddings)
        if device is not None:
            tensor = tensor.to(device)
        if len(tensor.shape) == 2:
            return tensor.unsqueeze(0)
        return tensor

    if isinstance(embeddings, torch.Tensor):
        if device is not None and embeddings.device != device:
            embeddings = embeddings.to(device)
        if len(embeddings.shape) == 2:
            return embeddings.unsqueeze(0)
        return embeddings

    if isinstance(embeddings, list) and isinstance(embeddings[0], torch.Tensor):
        if device is not None:
            return [embedding.to(device) for embedding in embeddings]
        return embeddings

    if isinstance(embeddings, list) and isinstance(embeddings[0], np.ndarray):
        if device is not None:
            return [torch.from_numpy(emb).to(device) for emb in embeddings]
        else:
            return [torch.from_numpy(emb) for emb in embeddings]
    
    if isinstance(embeddings, list):
        # Convert list to tensor
        tensor = torch.tensor(embeddings)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


class Flat(Base):
    """Flat index using exact nearest neighbor search with GPU acceleration.

    This is a simple in-memory index that performs exact search using dot product similarity.
    It's suitable for small to medium-sized datasets where exact results are needed.

    **Important Notes:**
    - This is an **exact** nearest neighbor search (not approximate)
    - All operations are in-memory (no persistence)
    - Supports GPU acceleration via PyTorch
    - Uses batched processing for efficient memory usage

    Parameters
    ----------
    name
        The name of the index collection.
    embedding_size
        The number of dimensions of the embeddings.
    device
        The device to use for computation ('cuda', 'cpu', or torch.device).
        Defaults to 'cuda' if available, otherwise 'cpu'.
    search_batch_size
        The batch size for processing query tokens during search.
        Larger values are faster but use more memory.
        Defaults to 1024.
    verbose
        Whether to enable verbose logging of timing and operations.
        Defaults to False for cleaner output.

    """

    def __init__(
        self,
        name: str | None = "Flat_index",
        embedding_size: int = 128,
        device: str | torch.device | None = None,
        search_batch_size: int = 1024,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.embedding_size = embedding_size
        self.verbose = verbose
        self.search_batch_size = search_batch_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        if self.verbose:
            logger.info(f"[Flat] Using device: {self.device}")

        # In-memory data structures only (no file I/O)
        self.all_embeddings = None
        self.embedding_id_to_position = {}  # Map embedding ID to position in array
        self.position_to_embedding_id = {}  # Reverse mapping: position to embedding ID
        self.embeddings_to_documents_ids = {}  # In-memory mapping: embedding ID -> document ID
        self.documents_ids_to_embeddings = {}  # In-memory mapping: document ID -> list of embedding IDs
        self._documents_added = False  # Track if documents have been added

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
                "Flat index only supports adding all documents at once. "
                "Documents have already been added."
            )

        if self.verbose:
            logger.info(f"[Flat] Adding {len(documents_ids)} documents to index...")

        # Reshape embeddings (convert to tensors on device)
        documents_embeddings = reshape_embeddings(embeddings=documents_embeddings, device=self.device)

        # Flatten all document embeddings at once
        if isinstance(documents_embeddings, list):
            flattened_embeddings = torch.cat([doc_emb for doc_emb in documents_embeddings], dim=0)
        else:
            # Already a tensor, flatten first two dimensions
            batch_size, n_tokens, emb_dim = documents_embeddings.shape
            flattened_embeddings = documents_embeddings.view(-1, emb_dim)

        # Ensure float32
        flattened_embeddings = flattened_embeddings.float()

        if self.verbose:
            logger.info(f"[Flat] Created embedding matrix with shape {flattened_embeddings.shape} on {flattened_embeddings.device}")

        # Assign embedding IDs sequentially starting from 0
        embedding_ids = list(range(len(flattened_embeddings)))

        # Store mappings in memory
        total = 0
        if isinstance(documents_embeddings, list):
            doc_lengths = [len(doc_emb) for doc_emb in documents_embeddings]
        else:
            doc_lengths = [documents_embeddings.shape[1]] * documents_embeddings.shape[0]
        
        for doc_id, num_tokens in zip(documents_ids, doc_lengths):
            document_embeddings_ids = embedding_ids[total : total + num_tokens]
            self.documents_ids_to_embeddings[doc_id] = document_embeddings_ids

            # Update in-memory mapping
            for emb_id in document_embeddings_ids:
                self.embeddings_to_documents_ids[str(emb_id)] = doc_id

            total += num_tokens

        # Set position mappings sequentially (embedding ID == position for simplicity)
        for pos, emb_id in enumerate(embedding_ids):
            self.embedding_id_to_position[emb_id] = pos
            self.position_to_embedding_id[pos] = emb_id

        # Store embeddings on device
        self.all_embeddings = flattened_embeddings

        # Mark that documents have been added
        self._documents_added = True

        if self.verbose:
            logger.info(f"[Flat] Successfully added {len(documents_ids)} documents ({len(flattened_embeddings)} embeddings)")

    def remove_documents(self, documents_ids: list[str]) -> None:
        """Remove documents from the index.
        
        Not supported for Flat index.

        Parameters
        ----------
        documents_ids
            The documents IDs to remove.

        Raises
        ------
        NotImplementedError
            Document removal is not supported for Flat index.

        """
        raise NotImplementedError(
            "Document removal is not supported for Flat index."
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
            Not yet implemented for Flat index.

        Raises
        ------
        NotImplementedError
            If subset is provided (not yet implemented).

        """
        if subset is not None:
            raise NotImplementedError(
                "Subset filtering is not yet implemented for Flat index."
            )

        if self.all_embeddings is None:
            raise ValueError("Index is empty, add documents before querying.")

        # Use in-memory mapping
        embeddings_to_documents_ids = self.embeddings_to_documents_ids

        # Reshape queries (convert to tensors on device)
        queries_embeddings = reshape_embeddings(embeddings=queries_embeddings, device=self.device)
        n_queries = len(queries_embeddings) if isinstance(queries_embeddings, list) else queries_embeddings.shape[0]

        if self.verbose:
            logger.info(f"[Flat] Processing {n_queries} queries")

        # Flatten query embeddings
        if isinstance(queries_embeddings, list):
            flattened_queries = torch.cat([q for q in queries_embeddings], dim=0)
        else:
            # Already a tensor, flatten first two dimensions
            batch_size, n_tokens, emb_dim = queries_embeddings.shape
            flattened_queries = queries_embeddings.view(-1, emb_dim)
        
        flattened_queries = flattened_queries.float()
        n_tokens_total = len(flattened_queries)

        if self.verbose:
            logger.info(f"[Flat] Computing dot products for {n_tokens_total} query tokens against {len(self.all_embeddings)} document embeddings")
            logger.info(f"[Flat] Using batch size: {self.search_batch_size}")

        # Process in batches to avoid OOM
        all_top_k_indices = []
        all_top_k_distances = []
        
        for batch_start in range(0, n_tokens_total, self.search_batch_size):
            batch_end = min(batch_start + self.search_batch_size, n_tokens_total)
            batch_queries = flattened_queries[batch_start:batch_end]
            
            # Compute dot products (exact search) for this batch
            # Shape: (batch_size, n_document_embeddings)
            with torch.no_grad():
                similarities = torch.mm(batch_queries, self.all_embeddings.T)
            
            # Get top-k indices and scores for each query token in batch
            top_k_distances, top_k_indices = torch.topk(similarities, k=min(k, similarities.shape[1]), dim=1, largest=True)
            
            all_top_k_indices.append(top_k_indices.cpu())
            all_top_k_distances.append(top_k_distances.cpu())
        
        # Concatenate all batches
        top_k_indices = torch.cat(all_top_k_indices, dim=0).numpy()
        top_k_distances = torch.cat(all_top_k_distances, dim=0).numpy()

        if self.verbose:
            logger.info(f"[Flat] Retrieved top-{k} results for each query token")

        # Map embedding indices back to document IDs
        if isinstance(queries_embeddings, list):
            n_tokens_per_query = [len(q) for q in queries_embeddings]
        else:
            n_tokens_per_query = [queries_embeddings.shape[1]] * queries_embeddings.shape[0]

        documents = []
        distances_list = []

        query_idx = 0
        for query_num, n_tokens in enumerate(n_tokens_per_query):
            query_documents = []
            query_distances = []

            for token_idx in range(n_tokens):
                token_neighbors = top_k_indices[query_idx]
                token_distances = top_k_distances[query_idx]

                # Map embedding indices to document IDs
                token_docs = []
                token_dists = []

                for neighbor_pos, dist in zip(token_neighbors, token_distances):
                    # neighbor_pos is a position in the array, map back to embedding ID
                    emb_id = self.position_to_embedding_id.get(int(neighbor_pos))

                    if emb_id is not None:
                        emb_id_str = str(emb_id)
                        doc_id = embeddings_to_documents_ids.get(emb_id_str)
                        if doc_id is not None:
                            token_docs.append(doc_id)
                            token_dists.append(float(dist))

                query_documents.append(token_docs[:k])
                query_distances.append(token_dists[:k])
                query_idx += 1

            documents.append(query_documents)
            distances_list.append(query_distances)

        if self.verbose:
            logger.info(f"[Flat] Completed retrieval for {n_queries} queries")

        return {
            "documents_ids": documents,
            "distances": np.array(distances_list),
        }

    def get_documents_embeddings(
        self, documents_ids: list[list[str]]
    ) -> list[list[torch.Tensor]]:
        """Retrieve document embeddings for re-ranking from Flat index.
        
        Returns list of lists of tensors, where each tensor has shape (seq_len, dim).
        """

        if self.all_embeddings is None:
            raise ValueError("Index is empty, add documents before retrieving embeddings.")

        # Retrieve embeddings from memory using their IDs
        reconstructed_embeddings = []
        for doc_group in documents_ids:
            group_embeddings = []
            for doc_id in doc_group:
                doc_embedding_ids = self.documents_ids_to_embeddings[doc_id]
                # Directly index into the tensor and stack - stays on device
                positions = [self.embedding_id_to_position[emb_id] for emb_id in doc_embedding_ids]
                doc_embeddings = self.all_embeddings[positions]  # Shape: (seq_len, dim)
                group_embeddings.append(doc_embeddings)
            reconstructed_embeddings.append(group_embeddings)

        return reconstructed_embeddings

