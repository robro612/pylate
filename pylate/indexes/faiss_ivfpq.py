from __future__ import annotations

import itertools
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from .base import Base
from .utils import log_memory

logger = logging.getLogger(__name__)


def _flatten_embeddings_to_fp32(
    documents_embeddings: list[torch.Tensor],
) -> np.ndarray:
    """Flatten a list of per-document tensors into a single (N, D) float32 numpy array.

    Parameters
    ----------
    documents_embeddings
        List of tensors, each with shape (n_tokens, embedding_dim).
        May be any dtype (fp32, fp16, bf16).

    Returns
    -------
    np.ndarray
        Float32 array with shape (total_tokens, embedding_dim).
    """
    parts = []
    for emb in documents_embeddings:
        if isinstance(emb, torch.Tensor):
            parts.append(emb.to(dtype=torch.float32, device="cpu").numpy())
        elif isinstance(emb, np.ndarray):
            parts.append(emb.astype(np.float32))
        else:
            parts.append(np.asarray(emb, dtype=np.float32))
    return np.concatenate(parts, axis=0)


def _reshape_queries(
    queries_embeddings: list | np.ndarray | torch.Tensor,
) -> list[np.ndarray]:
    """Normalise query embeddings into a list of 2-D float32 numpy arrays.

    Handles the various input formats the retriever may pass:
    - list[torch.Tensor]  (each shape (n_tokens, dim))
    - list[np.ndarray]
    - single torch.Tensor (batch, n_tokens, dim)
    - single np.ndarray   (batch, n_tokens, dim) or (n_tokens, dim)
    """
    if isinstance(queries_embeddings, torch.Tensor):
        arr = queries_embeddings.to(dtype=torch.float32, device="cpu").numpy()
        if arr.ndim == 2:
            return [arr]
        return [arr[i] for i in range(arr.shape[0])]

    if isinstance(queries_embeddings, np.ndarray):
        arr = queries_embeddings.astype(np.float32)
        if arr.ndim == 2:
            return [arr]
        return [arr[i] for i in range(arr.shape[0])]

    # list of tensors / arrays
    out = []
    for q in queries_embeddings:
        if isinstance(q, torch.Tensor):
            out.append(q.to(dtype=torch.float32, device="cpu").numpy())
        elif isinstance(q, np.ndarray):
            out.append(q.astype(np.float32))
        else:
            out.append(np.asarray(q, dtype=np.float32))
    return out


class FaissIVFPQ(Base):
    """Faiss IVFPQ index with GPU-accelerated training, adding, and search.

    This index uses Faiss ``IndexIVFPQ`` for compressed approximate nearest
    neighbour search.  Training and adding happen on GPU for speed; a CPU copy
    of the index is created once via :meth:`finalize` for ``reconstruct``
    (needed every query during ColBERT reranking) and ``save``.

    Lifecycle
    ---------
    1. ``train(documents_ids, documents_embeddings)`` -- trains IVFPQ on a
       random sample, then adds all provided vectors to the GPU index.
    2. ``add_documents(...)`` -- called zero or more times to add further
       batches (GPU only, no CPU sync).
    3. ``finalize()`` -- single ``index_gpu_to_cpu`` + ``make_direct_map``
       (~44 s at MSMARCO scale).  After this, search, reconstruct, and save
       are available.

    Parameters
    ----------
    name
        Collection name (used for persistence sub-directory).
    embedding_size
        Dimensionality of the embeddings (e.g. 128 for ColBERT).
    nlist
        Number of IVF Voronoi cells.  If ``None``, auto-tuned from the number
        of training vectors as ``min(4096, max(256, int(sqrt(n_train))))``.
    m
        Number of PQ sub-quantizers.  Must divide ``embedding_size``.
    nbits
        Bits per PQ code (usually 8).
    nprobe
        Number of cells visited at search time.
    training_sample_size
        Max vectors sampled from the first ``train()`` call for IVF/PQ
        training.
    device
        CUDA device ordinal (int) or ``"cpu"``.  Defaults to GPU 0 if
        available.
    index_folder
        Root directory for persistence.  ``None`` disables disk I/O.
    override
        If ``True``, ignore any existing index on disk and rebuild.
    verbose_level
        ``"none"`` / ``"init"`` / ``"all"``.
    """

    def __init__(
        self,
        name: str | None = "FaissIVFPQ_index",
        embedding_size: int = 128,
        nlist: Optional[int] = None,
        m: int = 64,
        nbits: int = 8,
        nprobe: int = 32,
        training_sample_size: int = 250_000,
        device: int | str | None = None,
        index_folder: str | None = None,
        override: bool = False,
        verbose_level: str = "none",
    ) -> None:
        self.name = name
        self.embedding_size = embedding_size
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.training_sample_size = training_sample_size
        self.index_folder = index_folder
        self.override = override
        self.verbose_level = verbose_level
        self.verbose = verbose_level in ("init", "all")

        # Resolve device
        if device is None:
            self._use_gpu = self._faiss_gpu_available()
            self._gpu_device = 0
        elif isinstance(device, int):
            self._use_gpu = True
            self._gpu_device = device
        elif device == "cpu":
            self._use_gpu = False
            self._gpu_device = 0
        else:
            self._use_gpu = True
            self._gpu_device = int(device)

        # Index state
        self.cpu_index = None  # CPU index for search / reconstruct / save
        self._gpu_resources = None

        # Lifecycle flags
        self._trained = False
        self._finalized = False

        # Document <-> position mappings
        self.doc_id_to_embedding_range: dict[str, tuple[int, int]] = {}
        self.position_to_doc_id: np.ndarray | None = None
        self._next_offset: int = 0

        # Accumulation buffer (stores tensors in original dtype until training)
        self._buffer_ids: list[str] = []
        self._buffer_embeddings: list[torch.Tensor] = []
        self._buffer_token_count: int = 0

        # Track original dtype for reconstruction (to match query dtype)
        self._original_dtype: torch.dtype | None = None

        # Try loading from disk
        if self.index_folder is not None and not self.override:
            idx_path = self._get_index_path()
            if idx_path is not None and (idx_path / "index.faiss").exists():
                self._load_index()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _faiss_gpu_available() -> bool:
        try:
            import faiss
            return faiss.get_num_gpus() > 0
        except Exception:
            return False

    @staticmethod
    def _import_faiss():
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "faiss is not installed.  Install faiss-gpu or faiss-cpu:\n"
                '  pip install faiss-gpu   # GPU support\n'
                '  pip install faiss-cpu   # CPU only'
            )

    def _log_retrieve(self) -> bool:
        return self.verbose_level == "all"

    def _get_index_path(self) -> Path | None:
        if self.index_folder is None or self.name is None:
            return None
        return Path(self.index_folder) / self.name

    def _get_gpu_resources(self):
        faiss = self._import_faiss()
        if self._gpu_resources is None:
            self._gpu_resources = faiss.StandardGpuResources()
        return self._gpu_resources

    def _to_gpu(self, cpu_index):
        """Move a CPU index to GPU with useFloat16 for PQ lookup tables."""
        faiss = self._import_faiss()
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        gpu_idx = faiss.index_cpu_to_gpu(
            self._get_gpu_resources(), self._gpu_device, cpu_index, cloner
        )
        return gpu_idx

    # ------------------------------------------------------------------
    # train (for backward compatibility)
    # ------------------------------------------------------------------

    def train(
        self,
        documents_ids: list[str],
        documents_embeddings: list[torch.Tensor],
    ) -> None:
        """Add documents to the index (calls ``add_documents`` internally).

        For backward compatibility. The index automatically trains when enough
        documents are accumulated.

        Parameters
        ----------
        documents_ids
            Document IDs for this batch.
        documents_embeddings
            List of tensors, each (n_tokens, dim).
        """
        self.add_documents(documents_ids, documents_embeddings)

    # ------------------------------------------------------------------
    # _train_and_offload
    # ------------------------------------------------------------------

    def _train_and_offload(self) -> None:
        """Train on accumulated buffer, add to GPU, immediately offload to CPU."""
        if self._trained:
            return

        faiss = self._import_faiss()
        train_start = time.time()

        # Flatten buffer to fp32 for Faiss
        flat = _flatten_embeddings_to_fp32(self._buffer_embeddings)
        n_vectors, dim = flat.shape
        assert dim == self.embedding_size, (
            f"Embedding dim mismatch: got {dim}, expected {self.embedding_size}"
        )

        if self.verbose:
            logger.info(
                f"[FaissIVFPQ] Training on {n_vectors:,} vectors (dim={dim})"
            )

        # Auto-tune nlist if not set
        if self.nlist is None:
            self.nlist = min(4096, max(256, int(math.sqrt(n_vectors))))
            if self.verbose:
                logger.info(f"[FaissIVFPQ] Auto nlist={self.nlist}")

        # Sample training vectors
        n_sample = min(self.training_sample_size, n_vectors)
        if n_sample < n_vectors:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_vectors, size=n_sample, replace=False)
            train_vectors = flat[indices]
        else:
            train_vectors = flat

        if self.verbose:
            logger.info(
                f"[FaissIVFPQ] Training with {len(train_vectors):,} sampled vectors"
            )

        # Build CPU index skeleton
        quantizer = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIVFPQ(
            quantizer, dim, self.nlist, self.m, self.nbits,
            faiss.METRIC_INNER_PRODUCT,
        )

        if self._use_gpu:
            # Train on GPU
            gpu_index = self._to_gpu(cpu_index)

            t0 = time.time()
            gpu_index.train(train_vectors)
            if self.verbose:
                logger.info(f"[FaissIVFPQ] GPU train: {time.time()-t0:.2f}s")

            gpu_index.nprobe = self.nprobe

            # Add buffered vectors on GPU
            t0 = time.time()
            gpu_index.add(flat)
            if self.verbose:
                logger.info(
                    f"[FaissIVFPQ] GPU add {n_vectors:,} vectors: {time.time()-t0:.2f}s"
                )

            # Immediately copy to CPU and free GPU
            t0 = time.time()
            self.cpu_index = faiss.index_gpu_to_cpu(gpu_index)
            del gpu_index
            self._gpu_resources = None
            if self.verbose:
                logger.info(
                    f"[FaissIVFPQ] Offloaded to CPU: {time.time()-t0:.2f}s"
                )
        else:
            # CPU-only path
            t0 = time.time()
            cpu_index.train(train_vectors)
            if self.verbose:
                logger.info(f"[FaissIVFPQ] CPU train: {time.time()-t0:.2f}s")

            cpu_index.nprobe = self.nprobe

            t0 = time.time()
            cpu_index.add(flat)
            if self.verbose:
                logger.info(
                    f"[FaissIVFPQ] CPU add {n_vectors:,} vectors: {time.time()-t0:.2f}s"
                )

            self.cpu_index = cpu_index

        self._trained = True

        # Update mappings for buffered documents
        self._update_mappings(self._buffer_ids, self._buffer_embeddings)

        # Clear buffer
        self._buffer_ids = []
        self._buffer_embeddings = []
        self._buffer_token_count = 0

        if self.verbose:
            logger.info(
                f"[FaissIVFPQ] Training complete: {time.time()-train_start:.2f}s  "
                f"ntotal={self.cpu_index.ntotal:,}"
            )

    # ------------------------------------------------------------------
    # add_documents
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents_ids: list[str],
        documents_embeddings: list[torch.Tensor],
    ) -> None:
        """Add documents to the index.

        If not yet trained, accumulates documents until reaching
        ``training_sample_size`` tokens, then trains on GPU and offloads to CPU.
        After training, adds directly to CPU index.

        Parameters
        ----------
        documents_ids
            Document IDs.
        documents_embeddings
            List of tensors, each (n_tokens, dim). Kept in original dtype.
        """
        if self._finalized:
            raise RuntimeError(
                "Cannot add documents after finalize(). "
                "Create a new index if you need to add more documents."
            )

        # Capture original dtype from first batch (for reconstruction)
        if self._original_dtype is None and len(documents_embeddings) > 0:
            first_emb = documents_embeddings[0]
            if isinstance(first_emb, torch.Tensor):
                self._original_dtype = first_emb.dtype
            else:
                # numpy array - convert to torch dtype equivalent
                import numpy as np
                if isinstance(first_emb, np.ndarray):
                    if first_emb.dtype == np.float16:
                        self._original_dtype = torch.float16
                    elif first_emb.dtype == np.float32:
                        self._original_dtype = torch.float32
                    else:
                        self._original_dtype = torch.float32  # default

        # Count tokens in this batch
        n_vectors = sum(emb.shape[0] for emb in documents_embeddings)

        if not self._trained:
            # Accumulate in buffer (keep original dtype to save memory)
            self._buffer_ids.extend(documents_ids)
            self._buffer_embeddings.extend(documents_embeddings)
            self._buffer_token_count += n_vectors

            if self.verbose:
                logger.info(
                    f"[FaissIVFPQ] Buffered {n_vectors:,} vectors "
                    f"({len(documents_ids)} docs). "
                    f"Total buffered: {self._buffer_token_count:,}"
                )

            # Auto-train when we hit threshold
            if self._buffer_token_count >= self.training_sample_size:
                if self.verbose:
                    logger.info(
                        f"[FaissIVFPQ] Reached {self._buffer_token_count:,} tokens, "
                        f"triggering training..."
                    )
                self._train_and_offload()
        else:
            # Already trained: add directly to CPU index
            flat = _flatten_embeddings_to_fp32(documents_embeddings)

            if self.verbose:
                logger.info(
                    f"[FaissIVFPQ] Adding {n_vectors:,} vectors "
                    f"({len(documents_ids)} docs) to CPU index..."
                )

            t0 = time.time()
            self.cpu_index.add(flat)
            if self.verbose:
                logger.info(
                    f"[FaissIVFPQ] CPU add: {time.time()-t0:.2f}s  "
                    f"ntotal={self.cpu_index.ntotal:,}"
                )

            self._update_mappings(documents_ids, documents_embeddings)

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Finalize the index for search / reconstruct / save.

        If documents are still buffered (not yet trained), trains on the buffer.
        Ensures the CPU index has a direct map for reconstruction.
        Must be called before ``__call__``, ``get_documents_embeddings``, or ``save``.
        """
        if self._finalized:
            raise RuntimeError("finalize() has already been called.")

        if self.verbose:
            logger.info("[FaissIVFPQ] Finalizing...")

        # If still buffered (never hit training threshold), train now
        if not self._trained:
            if self._buffer_token_count == 0:
                raise RuntimeError(
                    "No documents have been added. Call add_documents() first."
                )
            if self.verbose:
                logger.info(
                    f"[FaissIVFPQ] Training on buffered {self._buffer_token_count:,} "
                    f"tokens before finalize..."
                )
            self._train_and_offload()

        faiss = self._import_faiss()

        # Ensure direct map exists (needed for reconstruct)
        t0 = time.time()
        self.cpu_index.nprobe = self.nprobe
        self.cpu_index.make_direct_map()
        direct_map_time = time.time() - t0

        self._finalized = True

        if self.verbose:
            logger.info(
                f"[FaissIVFPQ] finalize: make_direct_map={direct_map_time:.2f}s  "
                f"ntotal={self.cpu_index.ntotal:,}"
            )

    # ------------------------------------------------------------------
    # __call__ (search)
    # ------------------------------------------------------------------

    def __call__(
        self,
        queries_embeddings: list[list[int | float]],
        k: int = 5,
        subset: list[list[str]] | list[str] | None = None,
    ) -> dict:
        """Search the index for nearest neighbours.

        Parameters
        ----------
        queries_embeddings
            Query embeddings (various formats accepted).
        k
            Number of nearest neighbours per query token.
        subset
            Not implemented.

        Returns
        -------
        dict
            ``{"documents_ids": [...], "distances": [...]}`` with the same
            nested structure as :class:`ScaNN`.
        """
        if subset is not None:
            raise NotImplementedError(
                "Subset filtering is not implemented for FaissIVFPQ."
            )

        # Auto-finalize on first retrieve (for compatibility with other indexes)
        if not self._finalized:
            if self.verbose:
                logger.info("[FaissIVFPQ] Auto-finalizing before first retrieve...")
            self.finalize()

        if self.cpu_index is None or self.cpu_index.ntotal == 0:
            raise ValueError("Index is empty, add documents before querying.")

        total_start = time.time()

        # Reshape queries
        queries = _reshape_queries(queries_embeddings)
        n_queries = len(queries)
        n_tokens_per_query = [len(q) for q in queries]

        # Flatten all query tokens
        flat_queries = np.concatenate(queries, axis=0).astype(np.float32)
        n_tokens_total = flat_queries.shape[0]

        if self._log_retrieve():
            logger.info(
                f"[FaissIVFPQ] Searching {n_tokens_total} tokens "
                f"(k={k}, {n_queries} queries)"
            )

        # Search on CPU index (no k limit, unlike GPU which caps at 2048)
        t0 = time.time()
        effective_k = min(k, self.cpu_index.ntotal)
        distances, neighbors = self.cpu_index.search(flat_queries, effective_k)
        search_time = time.time() - t0

        if self._log_retrieve():
            logger.info(
                f"[FaissIVFPQ] search: {search_time:.4f}s "
                f"({search_time/n_tokens_total*1000:.2f}ms/token)"
            )

        # Replace -1 neighbours (padding) with 0 distance
        mask = neighbors < 0
        if mask.any():
            distances[mask] = 0.0
            neighbors[mask] = 0

        # Map positions -> doc IDs  (vectorised)
        t0 = time.time()
        all_doc_ids = self.position_to_doc_id[neighbors]
        mapping_time = time.time() - t0

        # Reshape into nested structure: [query][token][k_neighbours]
        documents = []
        distances_list = []
        token_idx = 0
        for n_tokens in n_tokens_per_query:
            query_docs = []
            query_dists = []
            for _ in range(n_tokens):
                query_docs.append(all_doc_ids[token_idx])
                query_dists.append(distances[token_idx, :effective_k])
                token_idx += 1
            documents.append(query_docs)
            distances_list.append(query_dists)

        if self._log_retrieve():
            total_time = time.time() - total_start
            logger.info(
                f"[FaissIVFPQ] Total retrieval: {total_time:.4f}s "
                f"(search={search_time:.4f}s, mapping={mapping_time:.4f}s)"
            )

        return {
            "documents_ids": documents,
            "distances": distances_list,
        }

    # ------------------------------------------------------------------
    # get_documents_embeddings
    # ------------------------------------------------------------------

    def get_documents_embeddings(
        self, documents_ids: list[list[str]]
    ) -> list[list[np.ndarray]]:
        """Reconstruct document embeddings from the IVFPQ index.

        Uses the CPU index's ``reconstruct_n`` (PQ-approximate).

        Parameters
        ----------
        documents_ids
            Nested list: ``[[doc_id, ...], ...]``.

        Returns
        -------
        list[list[np.ndarray]]
            Each inner array has shape ``(seq_len, dim)``.
        """
        # Auto-finalize if needed (for compatibility with other indexes)
        if not self._finalized:
            if self.verbose:
                logger.info("[FaissIVFPQ] Auto-finalizing before reconstruction...")
            self.finalize()

        reconstructed = []
        for doc_group in documents_ids:
            group_embs = []
            for doc_id in doc_group:
                if doc_id not in self.doc_id_to_embedding_range:
                    raise ValueError(
                        f"Document ID '{doc_id}' not found in index."
                    )
                start, length = self.doc_id_to_embedding_range[doc_id]
                vecs = self.cpu_index.reconstruct_n(start, length)  # Always fp32 from Faiss

                # Convert back to original dtype to match query embeddings
                # This ensures dtype consistency for ColBERT scoring
                if self._original_dtype == torch.float16:
                    vecs = vecs.astype(np.float16)
                elif self._original_dtype == torch.bfloat16:
                    # numpy doesn't support bfloat16, convert via torch
                    vecs = torch.from_numpy(vecs).to(dtype=torch.bfloat16).numpy()
                # else: keep as fp32

                group_embs.append(vecs)
            reconstructed.append(group_embs)
        return reconstructed

    # ------------------------------------------------------------------
    # remove_documents (not supported)
    # ------------------------------------------------------------------

    def remove_documents(self, documents_ids: list[str]) -> None:
        """Not supported for FaissIVFPQ."""
        raise NotImplementedError(
            "Document removal is not supported for FaissIVFPQ index."
        )

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the index to disk."""
        # Auto-finalize if needed
        if not self._finalized:
            if self.verbose:
                logger.info("[FaissIVFPQ] Auto-finalizing before save...")
            self.finalize()

        faiss = self._import_faiss()
        idx_path = self._get_index_path()
        if idx_path is None:
            if self.verbose:
                logger.warning(
                    "[FaissIVFPQ] Cannot save: index_folder or name not set."
                )
            return

        idx_path.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            logger.info(f"[FaissIVFPQ] Saving index to {idx_path}...")

        t0 = time.time()

        # Write FAISS index (CPU copy)
        faiss.write_index(self.cpu_index, str(idx_path / "index.faiss"))

        # Write metadata
        # Convert torch dtype to string for JSON serialization
        dtype_str = None
        if self._original_dtype is not None:
            dtype_str = str(self._original_dtype).replace("torch.", "")

        metadata = {
            "embedding_size": self.embedding_size,
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "nprobe": self.nprobe,
            "training_sample_size": self.training_sample_size,
            "ntotal": self.cpu_index.ntotal,
            "original_dtype": dtype_str,
        }
        with open(idx_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Write doc_id -> (start, length) mapping
        with open(idx_path / "doc_id_to_embedding_range.tsv", "w") as f:
            for doc_id, (start, length) in self.doc_id_to_embedding_range.items():
                f.write(f"{doc_id}\t{start}\t{length}\n")

        if self.verbose:
            logger.info(
                f"[FaissIVFPQ] Saved in {time.time()-t0:.2f}s to {idx_path}"
            )

    def _load_index(self) -> None:
        """Load an existing index from disk.  Sets ``_finalized=True``."""
        faiss = self._import_faiss()
        idx_path = self._get_index_path()
        if idx_path is None:
            raise ValueError("Cannot load: index_folder or name not set.")

        if self.verbose:
            logger.info(f"[FaissIVFPQ] Loading index from {idx_path}...")

        t0 = time.time()

        # Read CPU index
        cpu_index = faiss.read_index(str(idx_path / "index.faiss"))
        cpu_index.make_direct_map()

        # Read metadata
        with open(idx_path / "metadata.json") as f:
            meta = json.load(f)
        self.embedding_size = meta.get("embedding_size", self.embedding_size)
        self.nlist = meta.get("nlist", self.nlist)
        self.m = meta.get("m", self.m)
        self.nbits = meta.get("nbits", self.nbits)
        self.nprobe = meta.get("nprobe", self.nprobe)
        self.training_sample_size = meta.get(
            "training_sample_size", self.training_sample_size
        )

        # Restore original dtype
        dtype_str = meta.get("original_dtype")
        if dtype_str:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            self._original_dtype = dtype_map.get(dtype_str, torch.float32)
        else:
            self._original_dtype = torch.float32  # default for old indexes

        # Read doc mappings
        mapping_path = idx_path / "doc_id_to_embedding_range.tsv"
        self.doc_id_to_embedding_range = {}
        with open(mapping_path) as f:
            for line in f:
                doc_id, start, length = line.strip().split("\t")
                self.doc_id_to_embedding_range[doc_id] = (
                    int(start),
                    int(length),
                )

        # Rebuild position_to_doc_id
        ntotal = cpu_index.ntotal
        self.position_to_doc_id = np.empty(ntotal, dtype=object)
        for doc_id, (start, length) in self.doc_id_to_embedding_range.items():
            self.position_to_doc_id[start : start + length] = doc_id
        self._next_offset = ntotal

        # Set nprobe on loaded index (search is always CPU)
        cpu_index.nprobe = self.nprobe
        self.cpu_index = cpu_index
        self._trained = True
        self._finalized = True

        # Clear buffer (loaded indexes bypass accumulation)
        self._buffer_ids = []
        self._buffer_embeddings = []
        self._buffer_token_count = 0

        if self.verbose:
            logger.info(
                f"[FaissIVFPQ] Loaded in {time.time()-t0:.2f}s  "
                f"ntotal={cpu_index.ntotal:,}  "
                f"docs={len(self.doc_id_to_embedding_range):,}"
            )

    # ------------------------------------------------------------------
    # Internal: update doc <-> position mappings
    # ------------------------------------------------------------------

    def _update_mappings(
        self,
        documents_ids: list[str],
        documents_embeddings: list[torch.Tensor],
    ) -> None:
        """Update ``doc_id_to_embedding_range`` and ``position_to_doc_id``."""
        doc_lengths = [emb.shape[0] for emb in documents_embeddings]
        total_new = sum(doc_lengths)
        new_total = self._next_offset + total_new

        # Grow position_to_doc_id
        if self.position_to_doc_id is None:
            self.position_to_doc_id = np.empty(new_total, dtype=object)
        elif new_total > len(self.position_to_doc_id):
            grown = np.empty(new_total, dtype=object)
            grown[: len(self.position_to_doc_id)] = self.position_to_doc_id
            self.position_to_doc_id = grown

        offset = self._next_offset
        for doc_id, n_tokens in zip(documents_ids, doc_lengths):
            self.doc_id_to_embedding_range[doc_id] = (offset, n_tokens)
            self.position_to_doc_id[offset : offset + n_tokens] = doc_id
            offset += n_tokens

        self._next_offset = offset
