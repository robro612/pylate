from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union, Collection
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np
from scipy.cluster import hierarchy
from tqdm.autonotebook import tqdm

try:
    import fastkmeans
except ImportError:
    fastkmeans = None

from ..utils import TokenTFIDFStats

# Type alias for compression artifacts
# Artifacts can be shape-matched (list[torch.Tensor]) or metadata (Any)
CompressionArtifacts = dict[str, Union[list[torch.Tensor], Any]]



class CompressionStrategyConfigBase(ABC):
    """Base class for compression strategy configurations.
    
    All compression strategy configs should inherit from this class.
    """
    
    @abstractmethod
    def serialize(self) -> dict:
        """Serialize this configuration to a JSON-compatible dictionary."""
        pass
    
    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """Return the strategy type identifier (e.g., 'idf_pruning', 'pooling')."""
        pass


class CompressionStrategy(ABC):
    """
    Base interface for compression strategies.
    
    Strategies apply compression to embeddings and maintain 1:1 mapping with shape-matched artifacts.
    Each strategy should declare which artifacts it requires via the `required_artifacts` class variable.
    
    Artifacts can be:
    - Shape-matched: `list[torch.Tensor]` - one tensor per document, must match embedding shape
    - Metadata: `Any` - corpus-level or document-level metadata (e.g., idf_stats)
    """
    
    required_artifacts: list[str] = []  # Class variable: list of artifact keys required by this strategy
    
    @abstractmethod
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply compression to embeddings and update artifacts.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain all required artifacts.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (compressed_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with compressed embeddings.
            Metadata artifacts are passed through unchanged.
        """
        pass
    
    def compress_parallel(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
        batch_size: int = 100,
        num_workers: Optional[int] = None,
        show_progress: bool = False,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply compression in parallel by batching documents and processing them concurrently.
        
        This method splits the documents into batches and processes each batch in parallel
        using threads. Results are then recombined to maintain the original order.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain all required artifacts.
        batch_size
            Number of documents to process in each batch. Defaults to 100.
        num_workers
            Number of worker threads to use. If None, defaults to min(batch_size, number of documents).
        show_progress
            If True, shows a progress bar during parallel compression. Defaults to False.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (compressed_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with compressed embeddings.
            Metadata artifacts are passed through unchanged.
        """
        if len(embeddings) == 0:
            return [], artifacts
        
        # Determine number of workers
        if num_workers is None:
            num_workers = min(batch_size, len(embeddings), 8)  # Cap at 8 workers by default
        num_workers = max(1, num_workers)  # At least 1 worker
        
        # If we have very few documents, just use the regular compress method
        # This avoids overhead of parallelization for small batches
        if len(embeddings) <= batch_size or len(embeddings) < num_workers * 2:
            return self.compress(embeddings, artifacts)
        
        # Split documents into batches
        batches = []
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            # Extract corresponding artifacts for this batch
            batch_artifacts = {}
            for artifact_name, artifact_value in artifacts.items():
                if isinstance(artifact_value, list):
                    batch_artifacts[artifact_name] = artifact_value[i:i + batch_size]
                else:
                    # Metadata artifacts are shared across all batches
                    batch_artifacts[artifact_name] = artifact_value
            batches.append((i, batch_embeddings, batch_artifacts))
        
        # Process batches in parallel
        all_compressed_embeddings = [None] * len(embeddings)
        all_updated_artifacts = {}
        
        # Initialize artifact structure
        for artifact_name, artifact_value in artifacts.items():
            if isinstance(artifact_value, list):
                all_updated_artifacts[artifact_name] = [None] * len(embeddings)
            else:
                # For metadata artifacts (non-list), preserve the original reference
                # This ensures shared artifacts like tfidf_stats are available to all batches
                all_updated_artifacts[artifact_name] = artifact_value
        
        def process_batch(start_idx: int, batch_emb: list[torch.Tensor], batch_art: CompressionArtifacts) -> tuple[int, list[torch.Tensor], CompressionArtifacts]:
            """Process a single batch and return results with start index."""
            # Add batch start index to artifacts for strategies that need global document indices
            # This allows strategies like IDFPruningStrategy to use correct doc_idx
            batch_art_with_start = batch_art.copy()
            batch_art_with_start["_batch_start_idx"] = start_idx
            compressed_emb, updated_art = self.compress(batch_emb, batch_art_with_start)
            # Remove the internal start_idx from returned artifacts
            if "_batch_start_idx" in updated_art:
                del updated_art["_batch_start_idx"]
            return start_idx, compressed_emb, updated_art
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(process_batch, start_idx, batch_emb, batch_art): start_idx
                for start_idx, batch_emb, batch_art in batches
            }
            
            # Collect results as they complete
            iterator = as_completed(future_to_batch) if not show_progress else tqdm(as_completed(future_to_batch), total=len(batches), desc="Parallel compression")
            for future in iterator:
                start_idx, compressed_emb, updated_art = future.result()
                
                # Store compressed embeddings
                for j, emb in enumerate(compressed_emb):
                    all_compressed_embeddings[start_idx + j] = emb
                
                # Store updated artifacts
                for artifact_name, artifact_value in updated_art.items():
                    if isinstance(artifact_value, list):
                        for j, artifact_item in enumerate(artifact_value):
                            all_updated_artifacts[artifact_name][start_idx + j] = artifact_item
                    else:
                        # For metadata artifacts (non-list), use the same reference
                        # This ensures shared artifacts like tfidf_stats are preserved
                        if all_updated_artifacts[artifact_name] is None:
                            all_updated_artifacts[artifact_name] = artifact_value
                        # If it's already set, keep the same reference (they should be identical)
        
        # Verify all embeddings were processed
        if None in all_compressed_embeddings:
            raise RuntimeError("Some batches failed to process during parallel compression")
        
        # Verify all shape-matched artifacts were processed
        for artifact_name, artifact_value in all_updated_artifacts.items():
            if isinstance(artifact_value, list) and None in artifact_value:
                raise RuntimeError(f"Some {artifact_name} artifacts failed to process during parallel compression")
        
        return all_compressed_embeddings, all_updated_artifacts

def validate_compression_artifact_shape(
    embeddings: list[torch.Tensor],
    artifact: dict[str, list[torch.Tensor]],
) -> None:
    """
    Validate that compression artifact shapes match embedding shapes.
    
    Parameters
    ----------
    embeddings
        List of embedding tensors (one per document)
    artifact
        List of artifact tensors (1-1 mapping to token embeddings)
    
    Raises
    ------
    ValueError
        If shapes don't match
    """
    
    for i, (emb, artifact) in enumerate(zip(embeddings, artifact)):
        # Embedding shape: [num_tokens, embedding_dim]
        # Artifact shape: [seq_len] or [seq_len, ...]
        # Artifacts should match the masked embeddings (same number of tokens)
        artifact_len = artifact.shape[0] if len(artifact.shape) > 0 else 0
        if artifact_len != emb.shape[0]:
            raise ValueError(
                f"Document {i}: artifact length ({artifact_len}) "
                f"does not match embedding tokens ({emb.shape[0]}). "
                f"Expected equal lengths."
            )


