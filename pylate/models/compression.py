from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Union, Collection
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch import nn
import numpy as np
from scipy.cluster import hierarchy
from tqdm.autonotebook import tqdm

try:
    import fastkmeans
except ImportError:
    fastkmeans = None

from .utils import TokenTFIDFStats

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


@dataclass
class IDFPruningConfig(CompressionStrategyConfigBase):
    """
    Configuration for IDF-based pruning.

    Prunes tokens with low IDF scores, i.e., tokens that occur frequently across documents
    (high document frequency) and are therefore less informative.

    Attributes
    ----------
    mode
        ``"global"`` identifies the k unique token types with the lowest IDF scores and prunes
        **all occurrences** of those k token types across all documents in the batch. Different
        documents may have different numbers of tokens removed depending on how many times those
        token types appear.
        ``"document"`` prunes k lowest-IDF token occurrences from each document independently.
    top_k
        Number of tokens (after the protected prefix) to prune/remove. Mutually exclusive with ``threshold``.
        Removes tokens with the **lowest IDF scores** (highest document frequency, less informative).
        In "global" mode, this is the number of unique token **types** to identify and prune all
        occurrences of across the entire batch.
        In "document" mode, this is the number of token **occurrences** to prune per document.
    threshold
        Prune tokens whose IDF score is < threshold. Mutually exclusive with ``top_k``.
        Lower IDF = more common across documents = less informative.
    protected_tokens
        Number of leading tokens to always retain (CLS / prefixes).
    use_tfidf
        If True, uses TF-IDF scoring (considers in-document frequency).
        If False, uses only IDF scoring. Defaults to False.

    track_pruned_tokens
        If True, tracks which tokens were pruned from each document. Access via
        ``strategy.get_pruned_tokens()`` after encoding. Defaults to False.
    show_progress_bar
        If True, shows a progress bar during pruning. Defaults to False.
    """

    mode: Literal["global", "document"] = "document"
    top_k: Optional[int] = None
    threshold: Optional[float] = None
    protected_tokens: int = 1
    use_tfidf: bool = False
    track_pruned_tokens: bool = False
    ignore_token_ids: Optional[Collection[int]] = None
    show_progress_bar: bool = False

    def __post_init__(self) -> None:
        if self.top_k is None and self.threshold is None:
            raise ValueError("IDF pruning requires either `top_k` or `threshold`.")
        if self.top_k is not None and self.threshold is not None:
            raise ValueError(
                "Provide only one of `top_k` or `threshold` for IDF pruning."
            )
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("`top_k` must be a positive integer.")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("`threshold` must be a finite float.")

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        result = {
            "mode": self.mode,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "protected_tokens": self.protected_tokens,
            "use_tfidf": self.use_tfidf,
            "track_pruned_tokens": self.track_pruned_tokens,
            "show_progress_bar": self.show_progress_bar,
        }
        # Convert set to list for JSON serialization
        if self.ignore_token_ids is not None:
            result["ignore_token_ids"] = list(self.ignore_token_ids)
        return result
    
    @property
    def strategy_type(self) -> str:
        return "idf_pruning"


@dataclass
class AttentionPruningConfig(CompressionStrategyConfigBase):
    """
    Configuration for attention-based pruning.

    Prunes tokens with low attention scores, i.e., tokens that receive less attention
    from the model and are therefore less important for the representation.

    Attributes
    ----------
    top_k
        Number of tokens (after the protected prefix) to prune/remove. Mutually exclusive with ``threshold``.
        Removes tokens with the **lowest attention scores** (least attended-to tokens).
        This is the number of token **occurrences** to prune per document.
    threshold
        Prune tokens whose attention score is < threshold. Mutually exclusive with ``top_k``.
        Lower attention = less important for the representation.
    head_reduction
        Reduction function for aggregating attention scores over heads: "sum" or "max".
        This parameter documents how attention scores were computed.
    protected_tokens
        Number of leading tokens to always retain (CLS / prefixes).
    track_pruned_tokens
        If True, tracks which tokens were pruned from each document. Access via
        ``strategy.get_pruned_tokens()`` after encoding. Defaults to False.
    show_progress_bar
        If True, shows a progress bar during pruning. Defaults to False.
    normalize_scores
        If True, normalizes attention scores with softmax after masking. This ensures
        scores sum to 1.0 per document. Defaults to False (use raw aggregated scores).
    """

    top_k: Optional[int] = None
    threshold: Optional[float] = None
    protected_tokens: int = 1
    track_pruned_tokens: bool = False
    show_progress_bar: bool = False
    normalize_scores: bool = False
    head_reduction: Literal["sum", "max"] = "sum"

    def __post_init__(self) -> None:
        if self.top_k is None and self.threshold is None:
            raise ValueError("Attention pruning requires either `top_k` or `threshold`.")
        if self.top_k is not None and self.threshold is not None:
            raise ValueError(
                "Provide only one of `top_k` or `threshold` for attention pruning."
            )
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("`top_k` must be a positive integer.")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("`threshold` must be a finite float.")

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "top_k": self.top_k,
            "threshold": self.threshold,
            "protected_tokens": self.protected_tokens,
            "track_pruned_tokens": self.track_pruned_tokens,
            "show_progress_bar": self.show_progress_bar,
            "normalize_scores": self.normalize_scores,
            "head_reduction": self.head_reduction,
        }
    
    @property
    def strategy_type(self) -> str:
        return "attention_pruning"


@dataclass
class CompactorPruningConfig(CompressionStrategyConfigBase):
    """
    Configuration for Compactor-based pruning that combines attention and leverage scores.

    Prunes tokens with low combined scores, where the score is computed as:
    (a - a_avg) / std(a) + lambda * (o - o_avg) / std(o)
    where a is attention scores and o is leverage scores.

    Attributes
    ----------
    top_k
        Number of tokens (after the protected prefix) to prune/remove. Mutually exclusive with ``threshold``.
        Removes tokens with the **lowest combined scores**.
        This is the number of token **occurrences** to prune per document.
    threshold
        Prune tokens whose combined score is < threshold. Mutually exclusive with ``top_k``.
        Lower combined score = less important for the representation.
    protected_tokens
        Number of leading tokens to always retain (CLS / prefixes).
    lambda_mix
        Mixing parameter for combining attention and leverage scores. Defaults to 1.0.
    sketch_dim
        Optional sketch dimension for leverage scores. If None, uses full dimension.
        This parameter is used when computing leverage scores (not in pruning itself).
    attention_head_reduction
        Reduction function for aggregating attention scores over heads: "sum" or "max".
        This parameter documents how attention scores were computed.
    leverage_head_reduction
        Reduction function for aggregating leverage scores over heads: "sum" or "max".
        This parameter documents how leverage scores were computed.
    track_pruned_tokens
        If True, tracks which tokens were pruned from each document. Access via
        ``strategy.get_pruned_tokens()`` after encoding. Defaults to False.
    show_progress_bar
        If True, shows a progress bar during pruning. Defaults to False.
    """

    top_k: Optional[int] = None
    threshold: Optional[float] = None
    protected_tokens: int = 1
    lambda_mix: float = 1.0
    sketch_dim: Optional[int] = None
    attention_head_reduction: Literal["sum", "max"] = "sum"
    leverage_head_reduction: Literal["sum", "max"] = "sum"
    track_pruned_tokens: bool = False
    show_progress_bar: bool = False

    def __post_init__(self) -> None:
        if self.top_k is None and self.threshold is None:
            raise ValueError("Compactor pruning requires either `top_k` or `threshold`.")
        if self.top_k is not None and self.threshold is not None:
            raise ValueError(
                "Provide only one of `top_k` or `threshold` for Compactor pruning."
            )
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("`top_k` must be a positive integer.")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("`threshold` must be a finite float.")
        if self.lambda_mix < 0:
            raise ValueError("`lambda_mix` must be non-negative.")

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "top_k": self.top_k,
            "threshold": self.threshold,
            "protected_tokens": self.protected_tokens,
            "lambda_mix": self.lambda_mix,
            "sketch_dim": self.sketch_dim,
            "attention_head_reduction": self.attention_head_reduction,
            "leverage_head_reduction": self.leverage_head_reduction,
            "track_pruned_tokens": self.track_pruned_tokens,
            "show_progress_bar": self.show_progress_bar,
        }
    
    @property
    def strategy_type(self) -> str:
        return "compactor_pruning"


@dataclass
class RandomPruningConfig(CompressionStrategyConfigBase):
    """
    Configuration for random pruning.
    
    Prunes tokens randomly, selecting k tokens to remove from each document.
    This is useful as a baseline for comparing other pruning strategies.
    
    Attributes
    ----------
    k
        Number of tokens (after the protected prefix) to randomly prune/remove per document.
    protected_tokens
        Number of leading tokens to always retain (CLS / prefixes).
    track_pruned_tokens
        If True, tracks which tokens were pruned from each document. Access via
        ``strategy.get_pruned_tokens()`` after encoding. Defaults to False.
    show_progress_bar
        If True, shows a progress bar during pruning. Defaults to False.
    """
    
    k: int
    protected_tokens: int = 1
    track_pruned_tokens: bool = False
    show_progress_bar: bool = False
    
    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("`k` must be a positive integer.")
    
    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.
        
        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "k": self.k,
            "protected_tokens": self.protected_tokens,
            "track_pruned_tokens": self.track_pruned_tokens,
            "show_progress_bar": self.show_progress_bar,
        }
    
    @property
    def strategy_type(self) -> str:
        return "random_pruning"


@dataclass
class PoolingConfig(CompressionStrategyConfigBase):
    pool_factor: int = 1
    protected_tokens: int = 1
    clustering_method: Literal["hierarchical", "spherical", "window", "random"] = "hierarchical"
    show_progress_bar: bool = False
    weight_by: Optional[Literal["attention", "leverage", "idf", "tfidf"]] = None
    attention_head_reduction: Literal["sum", "max"] = "sum"
    leverage_head_reduction: Literal["sum", "max"] = "sum"
    leverage_sketch_dim: Optional[int] = None
    stride: Optional[int] = None

    def __post_init__(self):
        """
        Validate configuration parameters.
        """
        if self.stride is not None and self.clustering_method != "window":
            raise ValueError(
                f"`stride` parameter is only valid for 'window' pooling method, "
                f"but clustering_method is '{self.clustering_method}'."
            )
        if self.stride is not None and self.stride <= 0:
            raise ValueError("`stride` must be a positive integer if provided.")

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        result = {
            "pool_factor": self.pool_factor,
            "protected_tokens": self.protected_tokens,
            "clustering_method": self.clustering_method,
            "show_progress_bar": self.show_progress_bar,
            "weight_by": self.weight_by,
        }
        if self.weight_by == "attention":
            result["attention_head_reduction"] = self.attention_head_reduction
        elif self.weight_by == "leverage":
            result["leverage_head_reduction"] = self.leverage_head_reduction
            result["leverage_sketch_dim"] = self.leverage_sketch_dim
        if self.stride is not None:
            result["stride"] = self.stride
        return result
    
    @property
    def strategy_type(self) -> str:
        return "pooling"


class CompressionStrategy(ABC):
    """
    Base interface for compression strategies.
    
    Strategies apply compression to embeddings and maintain 1:1 mapping with shape-matched artifacts.
    Each strategy should declare which artifacts it requires via the `required_artifacts` instance attribute.
    
    Artifacts can be:
    - Shape-matched: `list[torch.Tensor]` - one tensor per document, must match embedding shape
    - Metadata: `Any` - corpus-level or document-level metadata (e.g., idf_stats)
    """
    
    def __init__(self):
        """Initialize the compression strategy."""
        self.required_artifacts: list[str] = []  # Instance attribute: list of artifact keys required by this strategy
    
    def get_artifact_requirements(self) -> dict[str, dict[str, Any]]:
        """
        Get dictionary of artifact requirements with their hook creation arguments.
        
        This method returns a dictionary mapping artifact names to their hook creation arguments.
        For artifacts that don't need special arguments (like input_ids), an empty dict is returned.
        
        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping artifact names to their hook creation arguments.
            For example:
            - "attention_scores" -> {"head_reduction": "sum"}
            - "leverage_scores" -> {"sketch_dim": 64, "head_reduction": "sum"}
            - "input_ids" -> {} (no arguments needed)
        
        Notes
        -----
        Default implementation returns empty dicts for all required artifacts.
        Subclasses should override this if they need specific hook arguments.
        """
        return {artifact: {} for artifact in self.required_artifacts}
    
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


class IDFPruningStrategy(CompressionStrategy):
    """
    IDF-based pruning strategy that removes tokens with low IDF scores.
    
    This strategy computes TF-IDF statistics from input_ids and prunes tokens
    that occur frequently across documents (high document frequency, low IDF),
    which are less informative for retrieval.
    
    Supports two modes:
    - "global": Identifies low-scoring token types globally and prunes all occurrences
    - "document": Prunes low-scoring tokens independently per document
    """
    
    def __init__(self, config: IDFPruningConfig):
        """
        Initialize the IDF pruning strategy.
        
        Parameters
        ----------
        config
            IDF pruning configuration specifying mode, top_k/threshold, protected_tokens, etc.
        """
        super().__init__()
        self.config = config
        self.required_artifacts = ["input_ids"]  # Requires input_ids to compute TF-IDF stats
        self._pruned_tokens: Optional[list[list[int]]] = None  # Track pruned tokens if requested
    
    @property
    def name(self) -> str:
        """Name of this compression strategy."""
        mode_str = self.config.mode
        criterion_str = f"topk-{self.config.top_k}" if self.config.top_k else f"threshold-{self.config.threshold}"
        return f"idf_pruning_mode-{mode_str}_{criterion_str}"
    
    @property
    def strategy_type(self) -> str:
        """Strategy type identifier for serialization."""
        return "idf_pruning"
    
    def serialize(self) -> dict:
        """
        Serialize this strategy to a JSON-compatible dictionary.
        
        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "IDFPruningStrategy":
        """
        Create a strategy instance from a serialized dictionary.
        
        Parameters
        ----------
        data
            Dictionary containing serialized strategy data
            
        Returns
        -------
        IDFPruningStrategy
            Deserialized strategy instance
        """
        config_data = data.get("config", {})
        config = IDFPruningConfig(
            mode=config_data.get("mode", "document"),
            top_k=config_data.get("top_k"),
            threshold=config_data.get("threshold"),
            protected_tokens=config_data.get("protected_tokens", 1),
            use_tfidf=config_data.get("use_tfidf", False),
            track_pruned_tokens=config_data.get("track_pruned_tokens", False),
            ignore_token_ids=set(config_data.get("ignore_token_ids", [])) if config_data.get("ignore_token_ids") else None,
            show_progress_bar=config_data.get("show_progress_bar", False),
        )
        return cls(config)
    
    def get_pruned_tokens(self) -> Optional[list[list[int]]]:
        """
        Get the list of pruned tokens for each document.
        
        Returns
        -------
        Optional[list[list[int]]]
            List of pruned token IDs per document, or None if tracking is disabled.
            Only available if track_pruned_tokens=True was set in config.
        """
        return self._pruned_tokens
    
    def _compute_tfidf_stats(self, input_ids: list[torch.Tensor]) -> TokenTFIDFStats:
        """
        Compute TF-IDF statistics from input_ids.
        
        Parameters
        ----------
        input_ids
            List of input_id tensors (one per document)
        
        Returns
        -------
        TokenTFIDFStats
            Fitted TF-IDF statistics
        """
        # Convert tensors to lists for TokenTFIDFStats
        tokenized_docs = [doc_input_ids.cpu().tolist() for doc_input_ids in input_ids]
        
        # Compute TF-IDF statistics
        stats = TokenTFIDFStats(num_docs=len(tokenized_docs))
        stats.fit(tokenized_docs, show_progress=self.config.show_progress_bar)
        return stats
    
    def _get_token_score(self, stats: TokenTFIDFStats, doc_idx: int, token_id: int) -> float:
        """
        Get the score for a token (IDF or TF-IDF).
        
        Parameters
        ----------
        stats
            TF-IDF statistics
        doc_idx
            Document index
        token_id
            Token ID
        
        Returns
        -------
        float
            Token score (IDF or TF-IDF)
        """
        if self.config.use_tfidf:
            return stats.get_tfidf(doc_idx, token_id)
        else:
            return stats.get_idf(token_id)
    
    def _get_tokens_to_prune_global(
        self,
        stats: TokenTFIDFStats,
        input_ids: Optional[list[torch.Tensor]] = None,
    ) -> set[int]:
        """
        Identify token types to prune globally based on lowest scores.
        
        Parameters
        ----------
        stats
            TF-IDF statistics (computed on all documents)
        input_ids
            Optional list of input_id tensors. If provided, used to check protected tokens.
            If None, computes from stats alone (assumes no protected token filtering needed).
        
        Returns
        -------
        set[int]
            Set of token IDs to prune globally
        """
        # Collect all unique token types and their scores
        # Iterate over all tokens in the stats (from idf_scores which contains all tokens in corpus)
        token_scores: dict[int, float] = {}
        
        # Get all token IDs from stats
        all_token_ids = set(stats.idf_scores.keys())
        
        # If input_ids provided, we need to check protected tokens per document
        # Otherwise, just use IDF scores directly
        if input_ids is not None:
            # Check which tokens appear in protected positions
            tokens_in_protected_positions: set[int] = set()
            for doc_input_ids in input_ids:
                doc_tokens = doc_input_ids.cpu().tolist()
                protected_tokens = set(doc_tokens[:self.config.protected_tokens])
                tokens_in_protected_positions.update(protected_tokens)
            
            for token_id in all_token_ids:
                # Skip ignored tokens
                if self.config.ignore_token_ids and token_id in self.config.ignore_token_ids:
                    continue
                
                # Skip tokens that appear in protected positions (conservative approach)
                if token_id in tokens_in_protected_positions:
                    continue
                
                # For global mode with IDF, just use IDF score (doesn't depend on doc_idx)
                # For TF-IDF, we'd need to check all documents, but for now use IDF
                if self.config.use_tfidf:
                    # For TF-IDF, get minimum score across all documents where token appears
                    min_score = float('inf')
                    for doc_idx in range(stats.num_docs):
                        if token_id in stats.doc_token_counts[doc_idx]:
                            score = stats.get_tfidf(doc_idx, token_id)
                            min_score = min(min_score, score)
                    if min_score == float('inf'):
                        continue  # Token doesn't appear in any document
                    token_scores[token_id] = min_score
                else:
                    # Just use IDF score (same for all documents)
                    token_scores[token_id] = stats.get_idf(token_id)
        else:
            # No input_ids provided - compute from stats alone
            # Skip ignored tokens, but can't check protected tokens without input_ids
            for token_id in all_token_ids:
                if self.config.ignore_token_ids and token_id in self.config.ignore_token_ids:
                    continue
                
                # Use IDF score (or minimum TF-IDF if use_tfidf is True)
                if self.config.use_tfidf:
                    # For TF-IDF, get minimum score across all documents
                    min_score = float('inf')
                    for doc_idx in range(stats.num_docs):
                        if token_id in stats.doc_token_counts[doc_idx]:
                            score = stats.get_tfidf(doc_idx, token_id)
                            min_score = min(min_score, score)
                    if min_score == float('inf'):
                        continue
                    token_scores[token_id] = min_score
                else:
                    token_scores[token_id] = stats.get_idf(token_id)
        
        # Select tokens to prune based on top_k or threshold
        if self.config.top_k is not None:
            # Sort by score (ascending) and take top_k lowest
            sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1])
            num_to_prune = min(self.config.top_k, len(sorted_tokens))
            tokens_to_prune = {token_id for token_id, _ in sorted_tokens[:num_to_prune]}
        else:
            # Prune tokens with score < threshold
            tokens_to_prune = {
                token_id for token_id, score in token_scores.items()
                if score < self.config.threshold
            }
        
        return tokens_to_prune
    
    def _get_tokens_to_prune_document(
        self,
        stats: TokenTFIDFStats,
        doc_idx: int,
        doc_input_ids: torch.Tensor,
    ) -> set[int]:
        """
        Identify token occurrences to prune for a single document.
        
        Parameters
        ----------
        stats
            TF-IDF statistics
        doc_idx
            Document index
        doc_input_ids
            Input IDs tensor for this document
        
        Returns
        -------
        set[int]
            Set of token positions (indices) to prune in this document
        """
        doc_tokens = doc_input_ids.cpu().tolist()
        
        # Score each token occurrence (after protected tokens)
        token_scores: list[tuple[int, float]] = []  # (position, score)
        
        for pos in range(self.config.protected_tokens, len(doc_tokens)):
            token_id = doc_tokens[pos]
            
            # Skip ignored tokens
            if self.config.ignore_token_ids and token_id in self.config.ignore_token_ids:
                continue
            
            score = self._get_token_score(stats, doc_idx, token_id)
            token_scores.append((pos, score))
        
        # Select tokens to prune based on top_k or threshold
        if self.config.top_k is not None:
            # Sort by score (ascending) and take top_k lowest
            # Note: ignored tokens are already excluded from token_scores, so we'll prune exactly top_k non-ignored tokens
            sorted_tokens = sorted(token_scores, key=lambda x: x[1])
            # Take min(top_k, len(sorted_tokens)) to handle cases where we have fewer candidates than top_k
            num_to_prune = min(self.config.top_k, len(sorted_tokens))
            positions_to_prune = {pos for pos, _ in sorted_tokens[:num_to_prune]}
        else:
            # Prune tokens with score < threshold
            positions_to_prune = {
                pos for pos, score in token_scores
                if score < self.config.threshold
            }
        
        return positions_to_prune
    
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply IDF-based pruning to embeddings and update artifacts.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "input_ids" as a list of tensors.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        
        Raises
        ------
        ValueError
            If input_ids artifact is missing
        """
        # Get input_ids from artifacts
        if "input_ids" not in artifacts:
            raise ValueError(
                "IDFPruningStrategy requires 'input_ids' artifact. "
                "Ensure input_ids are provided when encoding."
            )
        
        input_ids = artifacts["input_ids"]
        if not isinstance(input_ids, list):
            raise ValueError("input_ids artifact must be a list of tensors")
        
        if len(input_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(input_ids)} input_ids but {len(embeddings)} embeddings"
            )
        
        # Validate all input_ids are tensors
        for idx, doc_input_ids in enumerate(input_ids):
            if not isinstance(doc_input_ids, torch.Tensor):
                raise ValueError(f"input_ids[{idx}] must be a torch.Tensor, got {type(doc_input_ids)}")
        
        # Initialize pruned tokens tracking if requested
        # Check if we're in parallel mode (pruned_tokens already pre-allocated)
        start_idx = artifacts.get("_batch_start_idx", 0)
        is_parallel_mode = (
            self.config.track_pruned_tokens and 
            self._pruned_tokens is not None and 
            isinstance(self._pruned_tokens, list) and
            len(self._pruned_tokens) > len(embeddings)  # Pre-allocated for parallel
        )
        
        if self.config.track_pruned_tokens and not is_parallel_mode:
            # Sequential mode: start fresh
            self._pruned_tokens = []
        elif not self.config.track_pruned_tokens:
            self._pruned_tokens = None
        
        # Compute or use cached TF-IDF statistics from artifacts
        if "tfidf_stats" in artifacts:
            stats = artifacts["tfidf_stats"]
        else:
            stats = self._compute_tfidf_stats(input_ids)
        
        # Determine tokens to prune based on mode
        # Store keep masks for artifact updates
        keep_masks: list[list[bool]] = []
        
        if self.config.mode == "global":
            # Global mode: identify token types to prune globally
            # Check if pre-computed tokens are available (from parallel processing)
            if "_tokens_to_prune_global" in artifacts:
                tokens_to_prune_global = artifacts["_tokens_to_prune_global"]
            else:
                # Sequential mode: compute from current input_ids
                tokens_to_prune_global = self._get_tokens_to_prune_global(stats, input_ids)
            
            # Prune all occurrences of these token types from all documents
            pruned_embeddings = []
            updated_input_ids = []
            
            criterion_str = f"top_k={self.config.top_k}" if self.config.top_k else f"threshold={self.config.threshold}"
            doc_pairs = list(zip(embeddings, input_ids))
            iterator = tqdm(
                enumerate(doc_pairs),
                desc=f"IDF pruning (global, {criterion_str})",
                total=len(embeddings),
                disable=not self.config.show_progress_bar,
            )
            
            for batch_doc_idx, (doc_embeddings, doc_input_ids) in iterator:
                doc_tokens = doc_input_ids.cpu().tolist()
                device = doc_input_ids.device
                dtype = doc_input_ids.dtype
                
                # Create mask: keep tokens that are not in tokens_to_prune_global
                # Always keep protected tokens
                keep_mask = []
                pruned_token_ids = []
                
                for pos, token_id in enumerate(doc_tokens):
                    if pos < self.config.protected_tokens:
                        # Always keep protected tokens
                        keep_mask.append(True)
                    elif token_id in tokens_to_prune_global:
                        # Prune this token
                        keep_mask.append(False)
                        if self.config.track_pruned_tokens:
                            pruned_token_ids.append(token_id)
                    else:
                        # Keep this token
                        keep_mask.append(True)
                
                # Apply mask to embeddings and input_ids
                # Ensure mask length matches embedding length
                if len(keep_mask) != doc_embeddings.shape[0]:
                    raise ValueError(
                        f"Mask length ({len(keep_mask)}) does not match embedding length "
                        f"({doc_embeddings.shape[0]}) for document {start_idx + batch_doc_idx}"
                    )
                keep_mask_tensor = torch.tensor(keep_mask, device=doc_embeddings.device, dtype=torch.bool)
                pruned_doc_embeddings = doc_embeddings[keep_mask_tensor]
                pruned_doc_tokens = torch.tensor(
                    [token for token, keep in zip(doc_tokens, keep_mask) if keep],
                    device=device,
                    dtype=dtype
                )
                
                pruned_embeddings.append(pruned_doc_embeddings)
                updated_input_ids.append(pruned_doc_tokens)
                
                if self.config.track_pruned_tokens:
                    if is_parallel_mode:
                        # Parallel mode: set at correct index
                        self._pruned_tokens[start_idx + batch_doc_idx] = pruned_token_ids
                    else:
                        # Sequential mode: append
                        self._pruned_tokens.append(pruned_token_ids)
                
                # Store keep mask for artifact updates
                keep_masks.append(keep_mask)
        
        else:  # document mode
            # Document mode: prune independently per document
            pruned_embeddings = []
            updated_input_ids = []
            
            criterion_str = f"top_k={self.config.top_k}" if self.config.top_k else f"threshold={self.config.threshold}"
            iterator = tqdm(
                zip(embeddings, input_ids),
                desc=f"IDF pruning (document, {criterion_str})",
                total=len(embeddings),
                disable=not self.config.show_progress_bar,
            )
            
            for batch_doc_idx, (doc_embeddings, doc_input_ids) in enumerate(iterator):
                # Use global document index for stats lookup
                # Get start index from artifacts if available (for parallel processing)
                start_idx = artifacts.get("_batch_start_idx", 0)
                global_doc_idx = start_idx + batch_doc_idx
                # Get positions to prune for this document
                positions_to_prune = self._get_tokens_to_prune_document(
                    stats, global_doc_idx, doc_input_ids
                )
                
                doc_tokens = doc_input_ids.cpu().tolist()
                device = doc_input_ids.device
                dtype = doc_input_ids.dtype
                
                # Create mask: keep tokens not in positions_to_prune
                keep_mask = []
                pruned_token_ids = []
                
                for pos, token_id in enumerate(doc_tokens):
                    if pos < self.config.protected_tokens:
                        # Always keep protected tokens
                        keep_mask.append(True)
                    elif pos in positions_to_prune:
                        # Prune this token
                        keep_mask.append(False)
                        if self.config.track_pruned_tokens:
                            pruned_token_ids.append(token_id)
                    else:
                        # Keep this token
                        keep_mask.append(True)
                
                # Apply mask to embeddings and input_ids
                # Ensure mask length matches embedding length
                if len(keep_mask) != doc_embeddings.shape[0]:
                    raise ValueError(
                        f"Mask length ({len(keep_mask)}) does not match embedding length "
                        f"({doc_embeddings.shape[0]}) for document {doc_idx}"
                    )
                keep_mask_tensor = torch.tensor(keep_mask, device=doc_embeddings.device, dtype=torch.bool)
                pruned_doc_embeddings = doc_embeddings[keep_mask_tensor]
                pruned_doc_tokens = torch.tensor(
                    [token for token, keep in zip(doc_tokens, keep_mask) if keep],
                    device=device,
                    dtype=dtype
                )
                
                pruned_embeddings.append(pruned_doc_embeddings)
                updated_input_ids.append(pruned_doc_tokens)
                
                if self.config.track_pruned_tokens:
                    if is_parallel_mode:
                        # Parallel mode: set at correct index
                        self._pruned_tokens[start_idx + batch_doc_idx] = pruned_token_ids
                    else:
                        # Sequential mode: append
                        self._pruned_tokens.append(pruned_token_ids)
                
                # Store keep mask for artifact updates
                keep_masks.append(keep_mask)
        
        # Update artifacts
        updated_artifacts = {}
        for artifact_name, artifact_value in artifacts.items():
            if artifact_name == "input_ids":
                # Update input_ids to match pruned embeddings
                updated_artifacts[artifact_name] = updated_input_ids
            elif isinstance(artifact_value, list):
                # Shape-matched artifact: apply same pruning mask
                pruned_artifacts = []
                for doc_idx, artifact_tokens in enumerate(artifact_value):
                    # Use the pre-computed keep mask
                    keep_mask = keep_masks[doc_idx]
                    
                    # Apply mask to artifact (assume tensor)
                    keep_mask_tensor = torch.tensor(keep_mask, device=artifact_tokens.device, dtype=torch.bool)
                    pruned_artifact = artifact_tokens[keep_mask_tensor]
                    
                    pruned_artifacts.append(pruned_artifact)
                
                updated_artifacts[artifact_name] = pruned_artifacts
            else:
                # Metadata artifact: pass through unchanged (including tfidf_stats)
                updated_artifacts[artifact_name] = artifact_value
        
        return pruned_embeddings, updated_artifacts
    
    def compress_parallel(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
        batch_size: int = 100,
        num_workers: Optional[int] = None,
        show_progress: bool = False,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply IDF-based pruning in parallel, computing global statistics first.
        
        This method computes TF-IDF statistics across all documents first (to ensure
        consistent results with sequential processing), then parallelizes the per-document
        pruning operations using the base class parallel implementation.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "input_ids" as a list of tensors.
        batch_size
            Number of documents to process in each batch. Defaults to 100.
        num_workers
            Number of worker threads to use. If None, defaults to min(batch_size, number of documents, 8).
        show_progress
            If True, shows a progress bar during parallel compression. Defaults to False.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        """
        # Get input_ids from artifacts to compute global stats
        if "input_ids" not in artifacts:
            raise ValueError(
                "IDFPruningStrategy requires 'input_ids' artifact. "
                "Ensure input_ids are provided when encoding."
            )
        
        input_ids = artifacts["input_ids"]
        if not isinstance(input_ids, list):
            raise ValueError("input_ids artifact must be a list of tensors")
        
        if len(input_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(input_ids)} input_ids but {len(embeddings)} embeddings"
            )
        
        # Compute TF-IDF statistics from ALL input_ids first (global computation)
        # This ensures results match sequential processing
        # Add stats to artifacts so compress() can use them
        # Make a copy of artifacts to avoid modifying the input
        if "tfidf_stats" in artifacts:
            stats = artifacts["tfidf_stats"]
        else:
            stats = self._compute_tfidf_stats(input_ids)

        artifacts_with_stats = artifacts.copy()
        artifacts_with_stats["tfidf_stats"] = stats
        
        # For global mode, pre-compute tokens_to_prune_global across ALL documents
        # This must be done before batching, otherwise each batch computes different tokens
        if self.config.mode == "global":
            tokens_to_prune_global = self._get_tokens_to_prune_global(stats, input_ids)
            artifacts_with_stats["_tokens_to_prune_global"] = tokens_to_prune_global
        
        # Initialize pruned tokens tracking if requested
        if self.config.track_pruned_tokens:
            self._pruned_tokens = [None] * len(embeddings)
        else:
            self._pruned_tokens = None
        
        # Call the base class compress_parallel, which will call compress() on each batch
        # compress() will use the stats and pre-computed tokens from artifacts
        result_emb, result_art = super().compress_parallel(embeddings, artifacts_with_stats, batch_size, num_workers, show_progress)
        
        # If tracking pruned tokens, ensure all were collected
        if self.config.track_pruned_tokens and self._pruned_tokens is not None:
            if None in self._pruned_tokens:
                raise RuntimeError("Some pruned tokens were not collected during parallel compression")
        
        return result_emb, result_art


class AttentionPruningStrategy(CompressionStrategy):
    """
    Attention-based pruning strategy that removes tokens with low attention scores.
    
    This strategy uses attention scores from the model's last layer to identify
    and prune tokens that receive less attention, which are less important for
    the document representation.
    """
    
    def __init__(self, config: AttentionPruningConfig):
        """
        Initialize the attention pruning strategy.
        
        Parameters
        ----------
        config
            Attention pruning configuration specifying top_k/threshold, protected_tokens, etc.
        """
        super().__init__()
        self.config = config
        self.required_artifacts = ["attention_scores"]  # Requires attention_scores to prune
        self._pruned_tokens: Optional[list[list[int]]] = None  # Track pruned tokens if requested
    
    @property
    def name(self) -> str:
        """Name of this compression strategy."""
        criterion_str = f"topk-{self.config.top_k}" if self.config.top_k else f"threshold-{self.config.threshold}"
        return f"attention_pruning_head_reduction-{self.config.head_reduction}_{criterion_str}"
    
    @property
    def strategy_type(self) -> str:
        """Strategy type identifier for serialization."""
        return "attention_pruning"
    
    def serialize(self) -> dict:
        """
        Serialize this strategy to a JSON-compatible dictionary.
        
        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AttentionPruningStrategy":
        """
        Create a strategy instance from a serialized dictionary.
        
        Parameters
        ----------
        data
            Dictionary containing serialized strategy data
            
        Returns
        -------
        AttentionPruningStrategy
            Deserialized strategy instance
        """
        config_data = data.get("config", {})
        config = AttentionPruningConfig(
            top_k=config_data.get("top_k"),
            threshold=config_data.get("threshold"),
            protected_tokens=config_data.get("protected_tokens", 1),
            track_pruned_tokens=config_data.get("track_pruned_tokens", False),
            show_progress_bar=config_data.get("show_progress_bar", False),
            normalize_scores=config_data.get("normalize_scores", False),
            head_reduction=config_data.get("head_reduction", "sum"),
        )
        return cls(config)

    def get_artifact_requirements(self) -> dict[str, dict[str, Any]]:
        """
        Get artifact requirements with hook creation arguments.
        
        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping artifact names to their hook creation arguments:
            - "attention_scores" -> {"head_reduction": self.config.head_reduction}
        """
        return {
            "attention_scores": {
                "head_reduction": self.config.head_reduction,
            },
        }
    
    def get_pruned_tokens(self) -> Optional[list[list[int]]]:
        """
        Get the list of pruned tokens for each document.
        
        Returns
        -------
        Optional[list[list[int]]]
            List of pruned token IDs per document, or None if tracking is disabled.
            Only available if track_pruned_tokens=True was set in config.
        """
        return self._pruned_tokens
    
    def _get_positions_to_prune(
        self,
        doc_attention_scores: torch.Tensor,
        doc_input_ids: torch.Tensor,
    ) -> set[int]:
        """
        Identify token positions to prune for a single document based on attention scores.
        
        Parameters
        ----------
        doc_attention_scores
            Attention scores tensor for this document, shape (seq_len,)
        doc_input_ids
            Input IDs tensor for this document, shape (seq_len,)
        
        Returns
        -------
        set[int]
            Set of token positions (indices) to prune in this document
        """
        doc_tokens = doc_input_ids.cpu().tolist()
        attention_scores = doc_attention_scores.cpu()
        
        # Normalize scores if requested
        if self.config.normalize_scores:
            # Apply softmax to normalize (after masking protected tokens if needed)
            # For now, we'll normalize all scores, but we could mask protected tokens first
            attention_scores = torch.softmax(attention_scores, dim=0)
        
        # Score each token occurrence (after protected tokens)
        token_scores: list[tuple[int, float]] = []  # (position, score)
        
        for pos in range(self.config.protected_tokens, len(doc_tokens)):
            score = attention_scores[pos].item()
            token_scores.append((pos, score))
        
        # Select tokens to prune based on top_k or threshold
        if self.config.top_k is not None:
            # Sort by score (ascending) and take top_k lowest
            sorted_tokens = sorted(token_scores, key=lambda x: x[1])
            # Take min(top_k, len(sorted_tokens)) to handle cases where we have fewer candidates than top_k
            num_to_prune = min(self.config.top_k, len(sorted_tokens))
            positions_to_prune = {pos for pos, _ in sorted_tokens[:num_to_prune]}
        else:
            # Prune tokens with score < threshold
            positions_to_prune = {
                pos for pos, score in token_scores
                if score < self.config.threshold
            }
        
        return positions_to_prune
    
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply attention-based pruning to embeddings and update artifacts.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "attention_scores" as a list of tensors.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        
        Raises
        ------
        ValueError
            If attention_scores artifact is missing
        """
        # Get attention_scores from artifacts
        if "attention_scores" not in artifacts:
            raise ValueError(
                "AttentionPruningStrategy requires 'attention_scores' artifact. "
                "Ensure attention_scores are provided when encoding."
            )
        
        attention_scores = artifacts["attention_scores"]
        if not isinstance(attention_scores, list):
            raise ValueError("attention_scores artifact must be a list of tensors")
        
        if len(attention_scores) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(attention_scores)} attention_scores but {len(embeddings)} embeddings"
            )
        
        # Validate all attention_scores are tensors
        for idx, doc_attention in enumerate(attention_scores):
            if not isinstance(doc_attention, torch.Tensor):
                raise ValueError(f"attention_scores[{idx}] must be a torch.Tensor, got {type(doc_attention)}")
            if len(doc_attention.shape) != 1:
                raise ValueError(f"attention_scores[{idx}] must be 1D, got shape {doc_attention.shape}")
        
        # Get input_ids if available (for tracking pruned tokens)
        input_ids = artifacts.get("input_ids")
        if input_ids is not None and not isinstance(input_ids, list):
            raise ValueError("input_ids artifact must be a list of tensors if provided")
        if input_ids is not None and len(input_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(input_ids)} input_ids but {len(embeddings)} embeddings"
            )
        
        # Initialize pruned tokens tracking if requested
        # Check if we're in parallel mode (pruned_tokens already pre-allocated)
        start_idx = artifacts.get("_batch_start_idx", 0)
        is_parallel_mode = (
            self.config.track_pruned_tokens and 
            self._pruned_tokens is not None and 
            isinstance(self._pruned_tokens, list) and
            len(self._pruned_tokens) > len(embeddings)  # Pre-allocated for parallel
        )
        
        if self.config.track_pruned_tokens and not is_parallel_mode:
            # Sequential mode: start fresh
            self._pruned_tokens = []
        elif not self.config.track_pruned_tokens:
            self._pruned_tokens = None
        
        # Prune tokens based on attention scores
        pruned_embeddings = []
        updated_attention_scores = []
        updated_input_ids = []
        
        criterion_str = f"top_k={self.config.top_k}" if self.config.top_k else f"threshold={self.config.threshold}"
        iterator = tqdm(
            enumerate(zip(embeddings, attention_scores)),
            desc=f"Attention pruning ({criterion_str})",
            total=len(embeddings),
            disable=not self.config.show_progress_bar,
        )
        
        for batch_doc_idx, (doc_embeddings, doc_attention_scores) in iterator:
            # Get input_ids for this document if available
            doc_input_ids = input_ids[batch_doc_idx] if input_ids is not None else None
            
            # Ensure attention scores match embedding length
            if doc_attention_scores.shape[0] != doc_embeddings.shape[0]:
                raise ValueError(
                    f"Document {start_idx + batch_doc_idx}: attention_scores length "
                    f"({doc_attention_scores.shape[0]}) does not match embedding length "
                    f"({doc_embeddings.shape[0]})"
                )
            
            # Get positions to prune
            if doc_input_ids is not None:
                positions_to_prune = self._get_positions_to_prune(doc_attention_scores, doc_input_ids)
            else:
                # If no input_ids, we can still prune based on attention scores alone
                # Create a dummy input_ids tensor for the function
                dummy_input_ids = torch.arange(doc_attention_scores.shape[0], device=doc_attention_scores.device)
                positions_to_prune = self._get_positions_to_prune(doc_attention_scores, dummy_input_ids)
            
            # Create mask: keep tokens not in positions_to_prune
            keep_mask = []
            pruned_token_ids = []
            
            for pos in range(doc_embeddings.shape[0]):
                if pos < self.config.protected_tokens:
                    # Always keep protected tokens
                    keep_mask.append(True)
                elif pos in positions_to_prune:
                    # Prune this token
                    keep_mask.append(False)
                    if self.config.track_pruned_tokens and doc_input_ids is not None:
                        pruned_token_ids.append(doc_input_ids[pos].item())
                else:
                    # Keep this token
                    keep_mask.append(True)
            
            # Apply mask to embeddings and attention scores
            keep_mask_tensor = torch.tensor(keep_mask, device=doc_embeddings.device, dtype=torch.bool)
            pruned_doc_embeddings = doc_embeddings[keep_mask_tensor]
            pruned_doc_attention = doc_attention_scores[keep_mask_tensor]
            
            pruned_embeddings.append(pruned_doc_embeddings)
            updated_attention_scores.append(pruned_doc_attention)
            
            # Update input_ids if available
            if doc_input_ids is not None:
                pruned_doc_tokens = doc_input_ids[keep_mask_tensor]
                updated_input_ids.append(pruned_doc_tokens)
            
            if self.config.track_pruned_tokens:
                if is_parallel_mode:
                    # Parallel mode: set at correct index
                    self._pruned_tokens[start_idx + batch_doc_idx] = pruned_token_ids
                else:
                    # Sequential mode: append
                    self._pruned_tokens.append(pruned_token_ids)
        
        # Update artifacts
        updated_artifacts = {}
        for artifact_name, artifact_value in artifacts.items():
            if artifact_name == "attention_scores":
                # Update attention_scores to match pruned embeddings
                updated_artifacts[artifact_name] = updated_attention_scores
            elif artifact_name == "input_ids" and input_ids is not None:
                # Update input_ids to match pruned embeddings
                updated_artifacts[artifact_name] = updated_input_ids
            elif isinstance(artifact_value, list):
                # Shape-matched artifact: apply same pruning mask
                pruned_artifacts = []
                for doc_idx, artifact_tokens in enumerate(artifact_value):
                    doc_attention_scores = attention_scores[doc_idx]
                    doc_input_ids_for_mask = input_ids[doc_idx] if input_ids is not None else None
                    
                    # Get positions to prune (same logic as above)
                    if doc_input_ids_for_mask is not None:
                        positions_to_prune = self._get_positions_to_prune(doc_attention_scores, doc_input_ids_for_mask)
                    else:
                        dummy_input_ids = torch.arange(doc_attention_scores.shape[0], device=doc_attention_scores.device)
                        positions_to_prune = self._get_positions_to_prune(doc_attention_scores, dummy_input_ids)
                    
                    # Create keep mask
                    keep_mask = []
                    for pos in range(len(artifact_tokens)):
                        if pos < self.config.protected_tokens:
                            keep_mask.append(True)
                        elif pos in positions_to_prune:
                            keep_mask.append(False)
                        else:
                            keep_mask.append(True)
                    
                    # Apply mask to artifact
                    keep_mask_tensor = torch.tensor(keep_mask, device=artifact_tokens.device, dtype=torch.bool)
                    pruned_artifact = artifact_tokens[keep_mask_tensor]
                    pruned_artifacts.append(pruned_artifact)
                
                updated_artifacts[artifact_name] = pruned_artifacts
            else:
                # Metadata artifact: pass through unchanged
                updated_artifacts[artifact_name] = artifact_value
        
        return pruned_embeddings, updated_artifacts
    
    def compress_parallel(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
        batch_size: int = 100,
        num_workers: Optional[int] = None,
        show_progress: bool = False,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply attention-based pruning in parallel.
        
        This method parallelizes the per-document pruning operations using the base
        class parallel implementation.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "attention_scores" as a list of tensors.
        batch_size
            Number of documents to process in each batch. Defaults to 100.
        num_workers
            Number of worker threads to use. If None, defaults to min(batch_size, number of documents, 8).
        show_progress
            If True, shows a progress bar during parallel compression. Defaults to False.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        """
        # Initialize pruned tokens tracking if requested
        if self.config.track_pruned_tokens:
            self._pruned_tokens = [None] * len(embeddings)
        else:
            self._pruned_tokens = None
        
        # Call the base class compress_parallel, which will call compress() on each batch
        result_emb, result_art = super().compress_parallel(embeddings, artifacts, batch_size, num_workers, show_progress)
        
        # If tracking pruned tokens, ensure all were collected
        if self.config.track_pruned_tokens and self._pruned_tokens is not None:
            if None in self._pruned_tokens:
                raise RuntimeError("Some pruned tokens were not collected during parallel compression")
        
        return result_emb, result_art


class CompactorPruningStrategy(CompressionStrategy):
    """
    Compactor-based pruning strategy that combines attention and leverage scores.
    
    This strategy uses both attention scores and leverage scores from the model to identify
    and prune tokens. The combined score is computed as:
    (a - a_avg) / std(a) + lambda * (o - o_avg) / std(o)
    where a is attention scores and o is leverage scores.
    """
    
    def __init__(self, config: CompactorPruningConfig):
        """
        Initialize the Compactor pruning strategy.
        
        Parameters
        ----------
        config
            Compactor pruning configuration specifying top_k/threshold, lambda_mix, etc.
        """
        super().__init__()
        self.config = config
        self.required_artifacts = ["attention_scores", "leverage_scores"]  # Requires both scores
        self._pruned_tokens: Optional[list[list[int]]] = None  # Track pruned tokens if requested
    
    @property
    def name(self) -> str:
        """Name of this compression strategy."""
        criterion_str = f"topk-{self.config.top_k}" if self.config.top_k else f"threshold-{self.config.threshold}"
        return f"compactor_pruning_lambda-{self.config.lambda_mix}_head_reduction-{self.config.attention_head_reduction}_sketch_dim-{self.config.sketch_dim}_{criterion_str}"
    
    @property
    def strategy_type(self) -> str:
        """Strategy type identifier for serialization."""
        return "compactor_pruning"
    
    def serialize(self) -> dict:
        """
        Serialize this strategy to a JSON-compatible dictionary.
        
        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CompactorPruningStrategy":
        """
        Create a strategy instance from a serialized dictionary.
        
        Parameters
        ----------
        data
            Dictionary containing serialized strategy data
            
        Returns
        -------
        CompactorPruningStrategy
            Deserialized strategy instance
        """
        config_data = data.get("config", {})
        config = CompactorPruningConfig(
            top_k=config_data.get("top_k"),
            threshold=config_data.get("threshold"),
            protected_tokens=config_data.get("protected_tokens", 1),
            lambda_mix=config_data.get("lambda_mix", 0.5),
            sketch_dim=config_data.get("sketch_dim"),
            attention_head_reduction=config_data.get("attention_head_reduction", "sum"),
            leverage_head_reduction=config_data.get("leverage_head_reduction", "sum"),
            track_pruned_tokens=config_data.get("track_pruned_tokens", False),
            show_progress_bar=config_data.get("show_progress_bar", True),
        )
        return cls(config)
    
    def get_artifact_requirements(self) -> dict[str, dict[str, Any]]:
        """
        Get artifact requirements with hook creation arguments.
        
        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping artifact names to their hook creation arguments:
            - "attention_scores" -> {"head_reduction": self.config.attention_head_reduction}
            - "leverage_scores" -> {"sketch_dim": self.config.sketch_dim, "head_reduction": self.config.leverage_head_reduction}
        """
        return {
            "attention_scores": {
                "head_reduction": self.config.attention_head_reduction,
            },
            "leverage_scores": {
                "sketch_dim": self.config.sketch_dim,
                "head_reduction": self.config.leverage_head_reduction,
            },
        }
    
    def get_pruned_tokens(self) -> Optional[list[list[int]]]:
        """
        Get the list of pruned tokens for each document.
        
        Returns
        -------
        Optional[list[list[int]]]
            List of pruned token IDs per document, or None if tracking is disabled.
            Only available if track_pruned_tokens=True was set in config.
        """
        return self._pruned_tokens
    
    def _compute_combined_scores(
        self,
        attention_scores: torch.Tensor,
        leverage_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined scores from attention and leverage scores.
        
        The formula is: (a - a_avg) / std(a) + lambda * (o - o_avg) / std(o)
        where a is attention scores and o is leverage scores.
        
        Parameters
        ----------
        attention_scores
            Attention scores tensor, shape (seq_len,)
        leverage_scores
            Leverage scores tensor, shape (seq_len,)
        
        Returns
        -------
        torch.Tensor
            Combined scores, shape (seq_len,)
        """
        # Normalize attention scores: (a - a_avg) / std(a)
        a_mean = attention_scores.mean()
        a_std = attention_scores.std()
        if a_std > 0:
            normalized_attention = (attention_scores - a_mean) / a_std
        else:
            # If std is 0, all values are the same, so normalized scores are 0
            normalized_attention = torch.zeros_like(attention_scores)
        
        # Normalize leverage scores: (o - o_avg) / std(o)
        o_mean = leverage_scores.mean()
        o_std = leverage_scores.std()
        if o_std > 0:
            normalized_leverage = (leverage_scores - o_mean) / o_std
        else:
            # If std is 0, all values are the same, so normalized scores are 0
            normalized_leverage = torch.zeros_like(leverage_scores)
        
        # Combine: normalized_attention + lambda * normalized_leverage
        combined_scores = normalized_attention + self.config.lambda_mix * normalized_leverage
        
        return combined_scores
    
    def _get_positions_to_prune(
        self,
        doc_attention_scores: torch.Tensor,
        doc_leverage_scores: torch.Tensor,
        doc_input_ids: torch.Tensor,
    ) -> set[int]:
        """
        Identify token positions to prune for a single document based on combined scores.
        
        Parameters
        ----------
        doc_attention_scores
            Attention scores tensor for this document, shape (seq_len,)
        doc_leverage_scores
            Leverage scores tensor for this document, shape (seq_len,)
        doc_input_ids
            Input IDs tensor for this document, shape (seq_len,)
        
        Returns
        -------
        set[int]
            Set of token positions (indices) to prune in this document
        """
        doc_tokens = doc_input_ids.cpu().tolist()
        attention_scores = doc_attention_scores.cpu()
        leverage_scores = doc_leverage_scores.cpu()
        
        # Compute combined scores
        combined_scores = self._compute_combined_scores(attention_scores, leverage_scores)
        
        # Score each token occurrence (after protected tokens)
        token_scores: list[tuple[int, float]] = []  # (position, score)
        
        for pos in range(self.config.protected_tokens, len(doc_tokens)):
            score = combined_scores[pos].item()
            token_scores.append((pos, score))
        
        # Select tokens to prune based on top_k or threshold
        if self.config.top_k is not None:
            # Sort by score (ascending) and take top_k lowest
            sorted_tokens = sorted(token_scores, key=lambda x: x[1])
            # Take min(top_k, len(sorted_tokens)) to handle cases where we have fewer candidates than top_k
            num_to_prune = min(self.config.top_k, len(sorted_tokens))
            positions_to_prune = {pos for pos, _ in sorted_tokens[:num_to_prune]}
        else:
            # Prune tokens with score < threshold
            positions_to_prune = {
                pos for pos, score in token_scores
                if score < self.config.threshold
            }
        
        return positions_to_prune
    
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply Compactor-based pruning to embeddings and update artifacts.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "attention_scores" and "leverage_scores" as lists of tensors.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        
        Raises
        ------
        ValueError
            If attention_scores or leverage_scores artifacts are missing
        """
        # Get attention_scores and leverage_scores from artifacts
        if "attention_scores" not in artifacts:
            raise ValueError(
                "CompactorPruningStrategy requires 'attention_scores' artifact. "
                "Ensure attention_scores are provided when encoding."
            )
        if "leverage_scores" not in artifacts:
            raise ValueError(
                "CompactorPruningStrategy requires 'leverage_scores' artifact. "
                "Ensure leverage_scores are provided when encoding."
            )
        
        attention_scores = artifacts["attention_scores"]
        leverage_scores = artifacts["leverage_scores"]
        
        if not isinstance(attention_scores, list):
            raise ValueError("attention_scores artifact must be a list of tensors")
        if not isinstance(leverage_scores, list):
            raise ValueError("leverage_scores artifact must be a list of tensors")
        
        if len(attention_scores) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(attention_scores)} attention_scores but {len(embeddings)} embeddings"
            )
        if len(leverage_scores) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(leverage_scores)} leverage_scores but {len(embeddings)} embeddings"
            )
        
        # Validate all scores are tensors
        for idx, doc_attention in enumerate(attention_scores):
            if not isinstance(doc_attention, torch.Tensor):
                raise ValueError(f"attention_scores[{idx}] must be a torch.Tensor, got {type(doc_attention)}")
            if len(doc_attention.shape) != 1:
                raise ValueError(f"attention_scores[{idx}] must be 1D, got shape {doc_attention.shape}")
        
        for idx, doc_leverage in enumerate(leverage_scores):
            if not isinstance(doc_leverage, torch.Tensor):
                raise ValueError(f"leverage_scores[{idx}] must be a torch.Tensor, got {type(doc_leverage)}")
            if len(doc_leverage.shape) != 1:
                raise ValueError(f"leverage_scores[{idx}] must be 1D, got shape {doc_leverage.shape}")
        
        # Get input_ids if available (for tracking pruned tokens)
        input_ids = artifacts.get("input_ids")
        if input_ids is not None and not isinstance(input_ids, list):
            raise ValueError("input_ids artifact must be a list of tensors if provided")
        if input_ids is not None and len(input_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(input_ids)} input_ids but {len(embeddings)} embeddings"
            )
        
        # Initialize pruned tokens tracking if requested
        # Check if we're in parallel mode (pruned_tokens already pre-allocated)
        start_idx = artifacts.get("_batch_start_idx", 0)
        is_parallel_mode = (
            self.config.track_pruned_tokens and 
            self._pruned_tokens is not None and 
            isinstance(self._pruned_tokens, list) and
            len(self._pruned_tokens) > len(embeddings)  # Pre-allocated for parallel
        )
        
        if self.config.track_pruned_tokens and not is_parallel_mode:
            # Sequential mode: start fresh
            self._pruned_tokens = []
        elif not self.config.track_pruned_tokens:
            self._pruned_tokens = None
        
        # Prune tokens based on combined scores
        pruned_embeddings = []
        updated_attention_scores = []
        updated_leverage_scores = []
        updated_input_ids = []
        
        criterion_str = f"top_k={self.config.top_k}" if self.config.top_k else f"threshold={self.config.threshold}"
        iterator = tqdm(
            enumerate(zip(embeddings, attention_scores, leverage_scores)),
            desc=f"Compactor pruning ({criterion_str})",
            total=len(embeddings),
            disable=not self.config.show_progress_bar,
        )
        
        for batch_doc_idx, (doc_embeddings, doc_attention_scores, doc_leverage_scores) in iterator:
            # Get input_ids for this document if available
            doc_input_ids = input_ids[batch_doc_idx] if input_ids is not None else None
            
            # Ensure scores match embedding length
            if doc_attention_scores.shape[0] != doc_embeddings.shape[0]:
                raise ValueError(
                    f"Document {start_idx + batch_doc_idx}: attention_scores length "
                    f"({doc_attention_scores.shape[0]}) does not match embedding length "
                    f"({doc_embeddings.shape[0]})"
                )
            if doc_leverage_scores.shape[0] != doc_embeddings.shape[0]:
                raise ValueError(
                    f"Document {start_idx + batch_doc_idx}: leverage_scores length "
                    f"({doc_leverage_scores.shape[0]}) does not match embedding length "
                    f"({doc_embeddings.shape[0]})"
                )
            
            # Get positions to prune
            if doc_input_ids is not None:
                positions_to_prune = self._get_positions_to_prune(
                    doc_attention_scores, doc_leverage_scores, doc_input_ids
                )
            else:
                # If no input_ids, we can still prune based on scores alone
                # Create a dummy input_ids tensor for the function
                dummy_input_ids = torch.arange(doc_attention_scores.shape[0], device=doc_attention_scores.device)
                positions_to_prune = self._get_positions_to_prune(
                    doc_attention_scores, doc_leverage_scores, dummy_input_ids
                )
            
            # Create mask: keep tokens not in positions_to_prune
            keep_mask = []
            pruned_token_ids = []
            
            for pos in range(doc_embeddings.shape[0]):
                if pos < self.config.protected_tokens:
                    # Always keep protected tokens
                    keep_mask.append(True)
                elif pos in positions_to_prune:
                    # Prune this token
                    keep_mask.append(False)
                    if self.config.track_pruned_tokens and doc_input_ids is not None:
                        pruned_token_ids.append(doc_input_ids[pos].item())
                else:
                    # Keep this token
                    keep_mask.append(True)
            
            # Apply mask to embeddings and scores
            keep_mask_tensor = torch.tensor(keep_mask, device=doc_embeddings.device, dtype=torch.bool)
            pruned_doc_embeddings = doc_embeddings[keep_mask_tensor]
            pruned_doc_attention = doc_attention_scores[keep_mask_tensor]
            pruned_doc_leverage = doc_leverage_scores[keep_mask_tensor]
            
            pruned_embeddings.append(pruned_doc_embeddings)
            updated_attention_scores.append(pruned_doc_attention)
            updated_leverage_scores.append(pruned_doc_leverage)
            
            # Update input_ids if available
            if doc_input_ids is not None:
                pruned_doc_tokens = doc_input_ids[keep_mask_tensor]
                updated_input_ids.append(pruned_doc_tokens)
            
            if self.config.track_pruned_tokens:
                if is_parallel_mode:
                    # Parallel mode: set at correct index
                    self._pruned_tokens[start_idx + batch_doc_idx] = pruned_token_ids
                else:
                    # Sequential mode: append
                    self._pruned_tokens.append(pruned_token_ids)
        
        # Update artifacts
        updated_artifacts = {}
        for artifact_name, artifact_value in artifacts.items():
            if artifact_name == "attention_scores":
                # Update attention_scores to match pruned embeddings
                updated_artifacts[artifact_name] = updated_attention_scores
            elif artifact_name == "leverage_scores":
                # Update leverage_scores to match pruned embeddings
                updated_artifacts[artifact_name] = updated_leverage_scores
            elif artifact_name == "input_ids" and input_ids is not None:
                # Update input_ids to match pruned embeddings
                updated_artifacts[artifact_name] = updated_input_ids
            elif isinstance(artifact_value, list):
                # Shape-matched artifact: apply same pruning mask
                pruned_artifacts = []
                for doc_idx, artifact_tokens in enumerate(artifact_value):
                    doc_attention_scores = attention_scores[doc_idx]
                    doc_leverage_scores = leverage_scores[doc_idx]
                    doc_input_ids_for_mask = input_ids[doc_idx] if input_ids is not None else None
                    
                    # Get positions to prune (same logic as above)
                    if doc_input_ids_for_mask is not None:
                        positions_to_prune = self._get_positions_to_prune(
                            doc_attention_scores, doc_leverage_scores, doc_input_ids_for_mask
                        )
                    else:
                        dummy_input_ids = torch.arange(doc_attention_scores.shape[0], device=doc_attention_scores.device)
                        positions_to_prune = self._get_positions_to_prune(
                            doc_attention_scores, doc_leverage_scores, dummy_input_ids
                        )
                    
                    # Create keep mask
                    keep_mask = []
                    for pos in range(len(artifact_tokens)):
                        if pos < self.config.protected_tokens:
                            keep_mask.append(True)
                        elif pos in positions_to_prune:
                            keep_mask.append(False)
                        else:
                            keep_mask.append(True)
                    
                    # Apply mask to artifact
                    keep_mask_tensor = torch.tensor(keep_mask, device=artifact_tokens.device, dtype=torch.bool)
                    pruned_artifact = artifact_tokens[keep_mask_tensor]
                    pruned_artifacts.append(pruned_artifact)
                
                updated_artifacts[artifact_name] = pruned_artifacts
            else:
                # Metadata artifact: pass through unchanged
                updated_artifacts[artifact_name] = artifact_value
        
        return pruned_embeddings, updated_artifacts
    
    def compress_parallel(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
        batch_size: int = 100,
        num_workers: Optional[int] = None,
        show_progress: bool = False,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply Compactor-based pruning in parallel.
        
        This method parallelizes the per-document pruning operations using the base
        class parallel implementation.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Must contain "attention_scores" and "leverage_scores" as lists of tensors.
        batch_size
            Number of documents to process in each batch. Defaults to 100.
        num_workers
            Number of worker threads to use. If None, defaults to min(batch_size, number of documents, 8).
        show_progress
            If True, shows a progress bar during parallel compression. Defaults to False.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        """
        # Initialize pruned tokens tracking if requested
        if self.config.track_pruned_tokens:
            self._pruned_tokens = [None] * len(embeddings)
        else:
            self._pruned_tokens = None
        
        # Call the base class compress_parallel, which will call compress() on each batch
        result_emb, result_art = super().compress_parallel(embeddings, artifacts, batch_size, num_workers, show_progress)
        
        # If tracking pruned tokens, ensure all were collected
        if self.config.track_pruned_tokens and self._pruned_tokens is not None:
            if None in self._pruned_tokens:
                raise RuntimeError("Some pruned tokens were not collected during parallel compression")
        
        return result_emb, result_art


class RandomPruningStrategy(CompressionStrategy):
    """
    Random pruning strategy that removes tokens randomly.
    
    This strategy randomly selects k tokens to prune from each document,
    which is useful as a baseline for comparing other pruning strategies.
    """
    
    def __init__(self, config: RandomPruningConfig):
        """
        Initialize the random pruning strategy.
        
        Parameters
        ----------
        config
            Random pruning configuration specifying k, protected_tokens, etc.
        """
        super().__init__()
        self.config = config
        self.required_artifacts = []  # No artifacts needed for random pruning
        self._pruned_tokens: Optional[list[list[int]]] = None  # Track pruned tokens if requested
    
    @property
    def name(self) -> str:
        """Name of this compression strategy."""
        return f"random_pruning_k-{self.config.k}"
    
    @property
    def strategy_type(self) -> str:
        """Strategy type identifier for serialization."""
        return "random_pruning"
    
    def serialize(self) -> dict:
        """
        Serialize this strategy to a JSON-compatible dictionary.
        
        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RandomPruningStrategy":
        """
        Create a strategy instance from a serialized dictionary.
        
        Parameters
        ----------
        data
            Dictionary containing serialized strategy data
            
        Returns
        -------
        RandomPruningStrategy
            Deserialized strategy instance
        """
        config_data = data.get("config", {})
        config = RandomPruningConfig(
            k=config_data.get("k"),
            protected_tokens=config_data.get("protected_tokens", 1),
            track_pruned_tokens=config_data.get("track_pruned_tokens", False),
            show_progress_bar=config_data.get("show_progress_bar", False),
        )
        return cls(config)
    
    def get_pruned_tokens(self) -> Optional[list[list[int]]]:
        """
        Get the list of pruned token IDs for each document.
        
        Returns
        -------
        Optional[list[list[int]]]
            List of pruned token ID lists (one per document), or None if tracking is disabled.
            Each inner list contains the token IDs that were pruned from that document.
        """
        return self._pruned_tokens
    
    def _get_positions_to_prune(
        self,
        doc_embeddings: torch.Tensor,
        doc_input_ids: Optional[torch.Tensor] = None,
    ) -> set[int]:
        """
        Randomly select positions to prune from a document.
        
        Parameters
        ----------
        doc_embeddings
            Embedding tensor for the document
        doc_input_ids
            Optional input IDs tensor for tracking pruned tokens
        
        Returns
        -------
        set[int]
            Set of positions (indices) to prune
        """
        num_tokens = doc_embeddings.shape[0]
        # Calculate how many tokens are available for pruning (excluding protected tokens)
        num_available = max(0, num_tokens - self.config.protected_tokens)
        
        if num_available == 0:
            return set()
        
        # Randomly select k positions from available tokens
        num_to_prune = min(self.config.k, num_available)
        
        # Generate random indices (after protected tokens)
        # Create a list of available positions
        available_indices = list(range(self.config.protected_tokens, num_tokens))
        # Randomly shuffle and take first num_to_prune
        perm = torch.randperm(len(available_indices), device=doc_embeddings.device)
        selected_indices = perm[:num_to_prune].cpu().tolist()
        # Convert to actual positions
        positions_to_prune = {available_indices[idx] for idx in selected_indices}
        
        return positions_to_prune
    
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply random pruning to embeddings and update artifacts.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Shape-matched artifacts will be updated to match
            the pruned embeddings. Metadata artifacts are passed through unchanged.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pruned_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pruned embeddings.
            Metadata artifacts are passed through unchanged.
        """
        # Get input_ids if available (for tracking pruned tokens)
        input_ids = artifacts.get("input_ids")
        if input_ids is not None and not isinstance(input_ids, list):
            raise ValueError("input_ids artifact must be a list of tensors if provided")
        if input_ids is not None and len(input_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(input_ids)} input_ids but {len(embeddings)} embeddings"
            )
        
        # Initialize pruned tokens tracking if requested
        # Check if we're in parallel mode (pruned_tokens already pre-allocated)
        start_idx = artifacts.get("_batch_start_idx", 0)
        is_parallel_mode = (
            self.config.track_pruned_tokens and 
            self._pruned_tokens is not None and 
            isinstance(self._pruned_tokens, list) and
            len(self._pruned_tokens) > len(embeddings)  # Pre-allocated for parallel
        )
        
        if self.config.track_pruned_tokens and not is_parallel_mode:
            # Sequential mode: start fresh
            self._pruned_tokens = []
        elif not self.config.track_pruned_tokens:
            self._pruned_tokens = None
        
        # Prune tokens randomly
        pruned_embeddings = []
        updated_input_ids = []
        
        iterator = tqdm(
            enumerate(embeddings),
            desc=f"Random pruning (k={self.config.k})",
            total=len(embeddings),
            disable=not self.config.show_progress_bar,
        )
        
        for batch_doc_idx, doc_embeddings in iterator:
            # Get input_ids for this document if available
            doc_input_ids = input_ids[batch_doc_idx] if input_ids is not None else None
            
            # Get positions to prune
            positions_to_prune = self._get_positions_to_prune(doc_embeddings, doc_input_ids)
            
            # Create mask: keep tokens not in positions_to_prune
            keep_mask = []
            pruned_token_ids = []
            
            for pos in range(doc_embeddings.shape[0]):
                if pos < self.config.protected_tokens:
                    # Always keep protected tokens
                    keep_mask.append(True)
                elif pos in positions_to_prune:
                    # Prune this token
                    keep_mask.append(False)
                    if self.config.track_pruned_tokens and doc_input_ids is not None:
                        pruned_token_ids.append(doc_input_ids[pos].item())
                else:
                    # Keep this token
                    keep_mask.append(True)
            
            # Apply mask to embeddings
            keep_mask_tensor = torch.tensor(keep_mask, device=doc_embeddings.device, dtype=torch.bool)
            pruned_doc_embeddings = doc_embeddings[keep_mask_tensor]
            
            pruned_embeddings.append(pruned_doc_embeddings)
            
            # Update input_ids if available
            if doc_input_ids is not None:
                pruned_doc_tokens = doc_input_ids[keep_mask_tensor]
                updated_input_ids.append(pruned_doc_tokens)
            
            if self.config.track_pruned_tokens:
                if is_parallel_mode:
                    # Parallel mode: set at correct index
                    self._pruned_tokens[start_idx + batch_doc_idx] = pruned_token_ids
                else:
                    # Sequential mode: append
                    self._pruned_tokens.append(pruned_token_ids)
        
        # Update artifacts
        updated_artifacts = {}
        for artifact_name, artifact_value in artifacts.items():
            if artifact_name == "input_ids" and input_ids is not None:
                # Update input_ids to match pruned embeddings
                updated_artifacts[artifact_name] = updated_input_ids
            elif isinstance(artifact_value, list):
                # Shape-matched artifact: apply same pruning mask
                pruned_artifacts = []
                for doc_idx, artifact_tokens in enumerate(artifact_value):
                    doc_embeddings_for_mask = embeddings[doc_idx]
                    
                    # Get positions to prune (same logic as above)
                    doc_input_ids_for_mask = input_ids[doc_idx] if input_ids is not None else None
                    positions_to_prune = self._get_positions_to_prune(
                        doc_embeddings_for_mask, doc_input_ids_for_mask
                    )
                    
                    # Create keep mask
                    keep_mask = []
                    for pos in range(len(artifact_tokens)):
                        if pos < self.config.protected_tokens:
                            keep_mask.append(True)
                        elif pos in positions_to_prune:
                            keep_mask.append(False)
                        else:
                            keep_mask.append(True)
                    
                    # Apply mask to artifact
                    keep_mask_tensor = torch.tensor(keep_mask, device=artifact_tokens.device, dtype=torch.bool)
                    pruned_artifact = artifact_tokens[keep_mask_tensor]
                    pruned_artifacts.append(pruned_artifact)
                
                updated_artifacts[artifact_name] = pruned_artifacts
            else:
                # Metadata artifact: pass through unchanged
                updated_artifacts[artifact_name] = artifact_value
        
        return pruned_embeddings, updated_artifacts


class PoolingStrategy(CompressionStrategy):
    """
    Pooling strategy that wraps the hierarchical/spherical pooling logic from ColBERT.
    
    This strategy pools embeddings by clustering similar token embeddings together
    and averaging them (optionally weighted by scores), reducing the number of tokens
    per document while preserving semantic information.
    
    The hierarchical method uses Ward's linkage clustering on cosine similarity distances.
    The spherical method uses fastkmeans clustering on the embeddings.
    
    When weight_by is specified, embeddings within each cluster are weighted by the
    specified score (attention, leverage, IDF, or TF-IDF) before averaging.
    """
    
    def __init__(self, config: PoolingConfig):
        """
        Initialize the pooling strategy.
        
        Parameters
        ----------
        config
            Pooling configuration specifying pool_factor, protected_tokens, clustering_method,
            and optionally weight_by for weighted pooling.
        """
        super().__init__()
        if config.pool_factor <= 0:
            raise ValueError("`pool_factor` must be a positive integer.")
        if config.protected_tokens < 0:
            raise ValueError("`protected_tokens` must be non-negative.")
        
        self.config = config
        
        # Set required artifacts based on weight_by
        if config.weight_by == "attention":
            self.required_artifacts = ["attention_scores"]
        elif config.weight_by == "leverage":
            self.required_artifacts = ["leverage_scores"]
        elif config.weight_by in ["idf", "tfidf"]:
            self.required_artifacts = ["input_ids"]
        else:
            self.required_artifacts = []  # No artifacts needed for unweighted pooling
    
    @property
    def name(self) -> str:
        """Name of this compression strategy."""
        weight_str = f"_weighted-{self.config.weight_by}" if self.config.weight_by else ""
        return f"pooling-{self.config.clustering_method}_k-{self.config.pool_factor}_p-{self.config.protected_tokens}{weight_str}"
    
    @property
    def strategy_type(self) -> str:
        """Strategy type identifier for serialization."""
        return "pooling"
    
    def serialize(self) -> dict:
        """
        Serialize this strategy to a JSON-compatible dictionary.
        
        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "type": self.strategy_type,
            "config": self.config.serialize(),
        }
    
    def get_artifact_requirements(self) -> dict[str, dict[str, Any]]:
        """
        Get artifact requirements with hook creation arguments.
        
        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping artifact names to their hook creation arguments.
            For attention/leverage scores, includes head_reduction and sketch_dim as needed.
            For input_ids (used by idf/tfidf), no arguments are needed.
        """
        if self.config.weight_by == "attention":
            return {
                "attention_scores": {
                    "head_reduction": self.config.attention_head_reduction,
                },
            }
        elif self.config.weight_by == "leverage":
            return {
                "leverage_scores": {
                    "sketch_dim": self.config.leverage_sketch_dim,
                    "head_reduction": self.config.leverage_head_reduction,
                },
            }
        elif self.config.weight_by in ["idf", "tfidf"]:
            return {
                "input_ids": {},  # input_ids doesn't need hook creation arguments
            }
        else:
            return {}
    
    @classmethod
    def from_dict(cls, data: dict) -> "PoolingStrategy":
        """
        Create a strategy instance from a serialized dictionary.
        
        Parameters
        ----------
        data
            Dictionary containing serialized strategy data
            
        Returns
        -------
        PoolingStrategy
            Deserialized strategy instance
        """
        config_data = data.get("config", {})
        config = PoolingConfig(
            pool_factor=config_data.get("pool_factor", 1),
            protected_tokens=config_data.get("protected_tokens", 1),
            clustering_method=config_data.get("clustering_method", "hierarchical"),
            show_progress_bar=config_data.get("show_progress_bar", False),
            weight_by=config_data.get("weight_by"),
            attention_head_reduction=config_data.get("attention_head_reduction", "sum"),
            leverage_head_reduction=config_data.get("leverage_head_reduction", "sum"),
            leverage_sketch_dim=config_data.get("leverage_sketch_dim"),
            stride=config_data.get("stride", config_data.get("pool_factor") if config_data.get("clustering_method") == "window" else None),
        )
        return cls(config)

    def _pool_embeddings_window(
        self,
        documents_embeddings: list[torch.Tensor],
        pool_factor: int,
        protected_tokens: int,
        weights: Optional[list[torch.Tensor]] = None,
        stride: Optional[int] = None,
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """
        Pools the embeddings by averaging sequential tokens using a sliding window approach.
        
        This method applies window-based pooling with a specified window size and stride.
        Protected tokens at the start of each document are preserved and not pooled.
        The pooling uses weighted averaging when weights are provided, and normalizes
        the resulting embeddings at the end.
        
        Parameters
        ----------
        documents_embeddings
            A list of embeddings for each document. Each tensor has shape (seq_len, embedding_dim).
        pool_factor
            Window size for pooling (number of tokens to average in each window).
        protected_tokens
            Number of tokens to protect from pooling at the start of each document.
        weights
            Optional list of weight tensors (one per document) for weighted pooling.
            Each weight tensor should have shape (seq_len,) matching the corresponding embeddings.
            If None, uses unweighted averaging.
        stride
            Stride for the sliding window. If None, defaults to pool_factor (no overlap).
            If stride < pool_factor, windows will overlap.
        
        Returns
        -------
        tuple[list[torch.Tensor], list[list[int]]]
            A tuple of (pooled_embeddings, window_assignments).
            pooled_embeddings: A list of pooled embeddings for each document.
            window_assignments: A list of window assignment lists, one per document.
                Each assignment list maps original token indices (after protected_tokens) to window IDs.
        """
        # Use pool_factor as window_size, and set stride default
        window_size = pool_factor
        if stride is None:
            stride = window_size
        
        # Determine device from first embedding (respect original device)
        # Only use CUDA if all embeddings are already on CUDA, otherwise use CPU
        if documents_embeddings:
            first_device = documents_embeddings[0].device
            # Use CUDA only if CUDA is available AND all embeddings are already on CUDA
            if torch.cuda.is_available() and first_device.type == "cuda":
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        
        pooled_embeddings = []
        window_assignments = []
        
        iterator = tqdm(
            zip(documents_embeddings, weights) if weights else documents_embeddings,
            desc=f"Window pooling (window={window_size}, stride={stride})",
            disable=not self.config.show_progress_bar,
            leave=False,
        )
        
        for item in iterator:
            if weights:
                document_embeddings, doc_weights = item
                doc_weights = doc_weights.to(device=device)
            else:
                document_embeddings = item
                doc_weights = None
            document_embeddings = document_embeddings.to(device=device)
            
            # Separate protected tokens from the rest
            # Ensure protected_tokens doesn't exceed document length to avoid CUDA asserts
            num_doc_tokens = document_embeddings.shape[0]
            actual_protected = min(protected_tokens, num_doc_tokens)
            protected_embeddings = document_embeddings[:actual_protected]
            embeddings_to_pool = document_embeddings[actual_protected:]
            
            num_embeddings = len(embeddings_to_pool)
            
            # If no embeddings to pool, just return protected embeddings
            if num_embeddings == 0:
                pooled_embeddings.append(protected_embeddings)
                window_assignments.append([])
                continue
            
            # Apply window pooling with stride
            pooled_window_embeddings = []
            window_assignment = []
            
            # Slide window over embeddings_to_pool
            start_idx = 0
            window_id = 0
            while start_idx < num_embeddings:
                end_idx = min(start_idx + window_size, num_embeddings)
                window_indices = torch.arange(start_idx, end_idx, device=device)
                
                if len(window_indices) > 0:
                    window_emb = embeddings_to_pool[window_indices]
                    
                    if doc_weights is not None:
                        # Weighted average: extract weights for this window
                        # Adjust indices to account for protected tokens offset
                        window_weights = doc_weights[actual_protected:][window_indices]
                        # Normalize weights to sum to 1 (with safety check for zero sum)
                        weight_sum = window_weights.sum()
                        if weight_sum > 0:
                            window_weights = window_weights / weight_sum
                            # Weighted average
                            window_embedding = (window_emb * window_weights.unsqueeze(1)).sum(dim=0)
                        else:
                            # Fallback to unweighted average if all weights are zero
                            window_embedding = window_emb.mean(dim=0)
                    else:
                        # Unweighted average
                        window_embedding = window_emb.mean(dim=0)
                    
                    pooled_window_embeddings.append(window_embedding)
                    
                    # Store window assignment: map each token in this window to the window_id (1-indexed)
                    for idx in window_indices.cpu().tolist():
                        window_assignment.append(window_id + 1)  # 1-indexed to match hierarchical
                    
                    window_id += 1
                
                # Move to next window
                start_idx += stride
            
            # Combine: protected embeddings first, then pooled windows
            if pooled_window_embeddings:
                pooled_tensor = torch.stack(pooled_window_embeddings)
                
                if actual_protected > 0:
                    final_embeddings = torch.cat([protected_embeddings, pooled_tensor], dim=0)
                else:
                    final_embeddings = pooled_tensor
            else:
                # No windows created, just return protected embeddings
                final_embeddings = protected_embeddings
            
            # Normalize the entire final tensor (L2 normalization)
            if final_embeddings.shape[0] > 0:
                final_embeddings = torch.nn.functional.normalize(
                    input=final_embeddings, p=2, dim=1
                )
            
            pooled_embeddings.append(final_embeddings)
            window_assignments.append(window_assignment)
        
        return pooled_embeddings, window_assignments
        
    def _pool_embeddings_random(
        self,
        documents_embeddings: list[torch.Tensor],
        pool_factor: int,
        protected_tokens: int,
        weights: Optional[list[torch.Tensor]] = None,
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """
        Pools the embeddings by randomly partitioning tokens into equal-sized pools.
        
        This method randomly partitions embeddings into pools of approximately equal size,
        then averages embeddings within each pool (optionally weighted). Each embedding
        is assigned to exactly one pool. Protected tokens at the start of each document
        are preserved and not pooled.
        
        Parameters
        ----------
        documents_embeddings
            A list of embeddings for each document. Each tensor has shape (seq_len, embedding_dim).
        pool_factor
            Factor to determine the number of pools. If there are N embeddings to pool,
            approximately N // pool_factor pools will be created, each with approximately
            pool_factor embeddings.
        protected_tokens
            Number of tokens to protect from pooling at the start of each document.
        weights
            Optional list of weight tensors (one per document) for weighted pooling.
            Each weight tensor should have shape (seq_len,) matching the corresponding embeddings.
            If None, uses unweighted averaging.
        
        Returns
        -------
        tuple[list[torch.Tensor], list[list[int]]]
            A tuple of (pooled_embeddings, pool_assignments).
            pooled_embeddings: A list of pooled embeddings for each document.
            pool_assignments: A list of pool assignment lists, one per document.
                Each assignment list maps original token indices (after protected_tokens) to pool IDs.
        """
        # Determine device from first embedding (respect original device)
        # Only use CUDA if all embeddings are already on CUDA, otherwise use CPU
        if documents_embeddings:
            first_device = documents_embeddings[0].device
            # Use CUDA only if CUDA is available AND all embeddings are already on CUDA
            if torch.cuda.is_available() and first_device.type == "cuda":
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        
        pooled_embeddings = []
        pool_assignments = []
        
        iterator = tqdm(
            zip(documents_embeddings, weights) if weights else documents_embeddings,
            desc=f"Random pooling (factor={pool_factor})",
            disable=not self.config.show_progress_bar,
            leave=False,
        )
        
        for item in iterator:
            if weights:
                document_embeddings, doc_weights = item
                doc_weights = doc_weights.to(device=device)
            else:
                document_embeddings = item
                doc_weights = None
            document_embeddings = document_embeddings.to(device=device)
            
            # Separate protected tokens from the rest
            # Ensure protected_tokens doesn't exceed document length to avoid CUDA asserts
            num_doc_tokens = document_embeddings.shape[0]
            actual_protected = min(protected_tokens, num_doc_tokens)
            protected_embeddings = document_embeddings[:actual_protected]
            embeddings_to_pool = document_embeddings[actual_protected:]
            
            num_embeddings = len(embeddings_to_pool)
            
            # If no embeddings to pool, just return protected embeddings
            if num_embeddings == 0:
                pooled_embeddings.append(protected_embeddings)
                pool_assignments.append([])
                continue
            
            # Calculate number of pools and pool sizes
            # We want approximately num_embeddings // pool_factor pools
            num_pools = max(num_embeddings // pool_factor, 1)
            
            # Create random permutation of indices
            indices = torch.randperm(num_embeddings, device=device)
            
            # Partition indices into pools of approximately equal size
            pool_indices_list = []
            pool_assignment = [-1] * num_embeddings  # Initialize with -1
            
            # Calculate how many embeddings per pool
            base_pool_size = num_embeddings // num_pools
            remainder = num_embeddings % num_pools
            
            # Distribute embeddings into pools
            current_idx = 0
            for pool_id in range(num_pools):
                # Some pools get one extra embedding if there's a remainder
                pool_size = base_pool_size + (1 if pool_id < remainder else 0)
                
                # Get indices for this pool
                pool_indices = indices[current_idx:current_idx + pool_size]
                pool_indices_list.append(pool_indices)
                
                # Update assignment mapping (map from original index to pool_id, 1-indexed)
                for orig_idx in pool_indices.cpu().tolist():
                    pool_assignment[orig_idx] = pool_id + 1  # 1-indexed to match hierarchical
                
                current_idx += pool_size
            
            # Pool embeddings within each pool (optionally weighted)
            pooled_document_embeddings = []
            for pool_id in range(num_pools):
                pool_indices = pool_indices_list[pool_id]
                if len(pool_indices) > 0:
                    pool_emb = embeddings_to_pool[pool_indices]
                    
                    if doc_weights is not None:
                        # Weighted average: extract weights for this pool
                        pool_weights = doc_weights[actual_protected:][pool_indices]
                        # Normalize weights to sum to 1 (with safety check for zero sum)
                        weight_sum = pool_weights.sum()
                        if weight_sum > 0:
                            pool_weights = pool_weights / weight_sum
                            # Weighted average
                            pool_embedding = (pool_emb * pool_weights.unsqueeze(1)).sum(dim=0)
                        else:
                            # Fallback to unweighted average if all weights are zero
                            pool_embedding = pool_emb.mean(dim=0)
                    else:
                        # Unweighted average
                        pool_embedding = pool_emb.mean(dim=0)
                    
                    pooled_document_embeddings.append(pool_embedding)
            
            # Combine: protected embeddings first, then pooled pools
            if pooled_document_embeddings:
                pooled_tensor = torch.stack(pooled_document_embeddings)
                
                if actual_protected > 0:
                    final_embeddings = torch.cat([protected_embeddings, pooled_tensor], dim=0)
                else:
                    final_embeddings = pooled_tensor
            else:
                # No pools created, just return protected embeddings
                final_embeddings = protected_embeddings
            
            # Normalize the entire final tensor (L2 normalization)
            if final_embeddings.shape[0] > 0:
                final_embeddings = torch.nn.functional.normalize(
                    input=final_embeddings, p=2, dim=1
                )
            
            pooled_embeddings.append(final_embeddings)
            pool_assignments.append(pool_assignment)
        
        return pooled_embeddings, pool_assignments
        
    def _pool_embeddings_hierarchical(
        self,
        documents_embeddings: list[torch.Tensor],
        pool_factor: int,
        protected_tokens: int,
        weights: Optional[list[torch.Tensor]] = None,
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """
        Pools the embeddings hierarchically by clustering and averaging them (optionally weighted).
        
        This method wraps the exact same logic as ColBERT.pool_embeddings_hierarchical,
        with optional weighted averaging when weights are provided.
        
        Parameters
        ----------
        documents_embeddings
            A list of embeddings for each document.
        pool_factor
            Factor to determine the number of clusters.
        protected_tokens
            Number of tokens to protect from pooling at the start of each document.
        weights
            Optional list of weight tensors (one per document) for weighted pooling.
            Each weight tensor should have shape (seq_len,) matching the corresponding embeddings.
            If None, uses unweighted averaging.
        
        Returns
        -------
        tuple[list[torch.Tensor], list[list[int]]]
            A tuple of (pooled_embeddings, cluster_assignments).
            pooled_embeddings: A list of pooled embeddings for each document.
            cluster_assignments: A list of cluster assignment lists, one per document.
                Each assignment list maps original token indices (after protected_tokens) to cluster IDs.
        """
        # Determine device from first embedding (respect original device)
        # Only use CUDA if all embeddings are already on CUDA, otherwise use CPU
        if documents_embeddings:
            first_device = documents_embeddings[0].device
            # Use CUDA only if CUDA is available AND all embeddings are already on CUDA
            if torch.cuda.is_available() and first_device.type == "cuda":
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        
        pooled_embeddings = []
        cluster_assignments = []
        
        iterator = tqdm(
            zip(documents_embeddings, weights) if weights else documents_embeddings,
            desc=f"Hierarchical pooling (factor={pool_factor})",
            disable=not self.config.show_progress_bar,
            leave=False,
        )
        
        for item in iterator:
            if weights:
                document_embeddings, doc_weights = item
                doc_weights = doc_weights.to(device=device)
            else:
                document_embeddings = item
                doc_weights = None
            document_embeddings = document_embeddings.to(device=device)
            
            # Separate protected tokens from the rest
            # Ensure protected_tokens doesn't exceed document length to avoid CUDA asserts
            num_doc_tokens = document_embeddings.shape[0]
            actual_protected = min(protected_tokens, num_doc_tokens)
            protected_embeddings = document_embeddings[:actual_protected]
            embeddings_to_pool = document_embeddings[actual_protected:]
            
            num_embeddings = len(embeddings_to_pool)
            
            # If no embeddings to pool, just return protected embeddings
            if num_embeddings == 0:
                pooled_embeddings.append(protected_embeddings)
                cluster_assignments.append([])
                continue
            
            # Compute cosine similarity and convert to distance matrix
            cosine_similarities = torch.mm(
                input=embeddings_to_pool, mat2=embeddings_to_pool.t()
            )
            distance_matrix = 1 - cosine_similarities.cpu().numpy()
            
            # Perform hierarchical clustering using Ward's method
            clusters = hierarchy.linkage(distance_matrix, method="ward")
            
            # Determine the number of clusters based on pool_factor
            num_clusters = max(num_embeddings // pool_factor, 1)
            cluster_labels = hierarchy.fcluster(
                clusters, t=num_clusters, criterion="maxclust"
            )
            
            # Store cluster assignments for artifact mapping
            cluster_assignments.append(cluster_labels.tolist())
            
            # Pool embeddings within each cluster (optionally weighted)
            pooled_document_embeddings = []
            for cluster_id in range(1, num_clusters + 1):
                cluster_indices = torch.where(
                    condition=torch.tensor(
                        data=cluster_labels == cluster_id, device=device
                    )
                )[0]
                if cluster_indices.numel() > 0:
                    cluster_emb = embeddings_to_pool[cluster_indices]
                    if doc_weights is not None:
                        # Weighted average: extract weights for this cluster
                        cluster_weights = doc_weights[actual_protected:][cluster_indices]
                        # Normalize weights to sum to 1 (with safety check for zero sum)
                        weight_sum = cluster_weights.sum()
                        if weight_sum > 0:
                            cluster_weights = cluster_weights / weight_sum
                            # Weighted average
                            cluster_embedding = (cluster_emb * cluster_weights.unsqueeze(1)).sum(dim=0)
                        else:
                            # Fallback to unweighted average if all weights are zero
                            cluster_embedding = cluster_emb.mean(dim=0)
                    else:
                        # Unweighted average
                        cluster_embedding = cluster_emb.mean(dim=0)
                    pooled_document_embeddings.append(cluster_embedding)
            
            # Re-append protected embeddings
            pooled_document_embeddings.extend(protected_embeddings)
            pooled_embeddings.append(torch.stack(tensors=pooled_document_embeddings))
        
        return pooled_embeddings, cluster_assignments
    
    def _pool_embeddings_spherical(
        self,
        documents_embeddings: list[torch.Tensor],
        pool_factor: int,
        protected_tokens: int,
        weights: Optional[list[torch.Tensor]] = None,
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """
        Pools the embeddings using spherical clustering via fastkmeans (optionally weighted).
        
        This method uses fastkmeans to perform k-means clustering on the embeddings,
        then averages embeddings within each cluster (optionally weighted) to create pooled representations.
        
        Parameters
        ----------
        documents_embeddings
            A list of embeddings for each document.
        pool_factor
            Factor to determine the number of clusters.
        protected_tokens
            Number of tokens to protect from pooling at the start of each document.
        weights
            Optional list of weight tensors (one per document) for weighted pooling.
            Each weight tensor should have shape (seq_len,) matching the corresponding embeddings.
            If None, uses unweighted averaging.
        
        Returns
        -------
        tuple[list[torch.Tensor], list[list[int]]]
            A tuple of (pooled_embeddings, cluster_assignments).
            pooled_embeddings: A list of pooled embeddings for each document.
            cluster_assignments: A list of cluster assignment lists, one per document.
                Each assignment list maps original token indices (after protected_tokens) to cluster IDs.
        
        Raises
        ------
        ImportError
            If fastkmeans is not installed
        """
        if fastkmeans is None:
            raise ImportError(
                "fastkmeans is required for spherical clustering. "
                "Install it with: pip install fastkmeans"
            )
        
        # Determine device from first embedding (respect original device)
        # Only use CUDA if all embeddings are already on CUDA, otherwise use CPU
        if documents_embeddings:
            first_device = documents_embeddings[0].device
            # Use CUDA only if CUDA is available AND all embeddings are already on CUDA
            if torch.cuda.is_available() and first_device.type == "cuda":
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        
        pooled_embeddings = []
        cluster_assignments = []
        
        iterator = tqdm(
            zip(documents_embeddings, weights) if weights else documents_embeddings,
            desc=f"Spherical pooling (factor={pool_factor})",
            disable=not self.config.show_progress_bar,
            leave=False,
        )
        
        for item in iterator:
            if weights:
                document_embeddings, doc_weights = item
                doc_weights = doc_weights.to(device=device)
            else:
                document_embeddings = item
                doc_weights = None
            document_embeddings = document_embeddings.to(device=device)
            
            # Separate protected tokens from the rest
            # Ensure protected_tokens doesn't exceed document length to avoid CUDA asserts
            num_doc_tokens = document_embeddings.shape[0]
            actual_protected = min(protected_tokens, num_doc_tokens)
            protected_embeddings = document_embeddings[:actual_protected]
            embeddings_to_pool = document_embeddings[actual_protected:]
            
            num_embeddings = len(embeddings_to_pool)
            
            # If no embeddings to pool, just return protected embeddings
            if num_embeddings == 0:
                if protected_tokens > 0:
                    pooled_embeddings.append(protected_embeddings)
                else:
                    # Empty document case
                    pooled_embeddings.append(torch.empty((0, document_embeddings.shape[1]), device=device))
                cluster_assignments.append([])
                continue
            
            # Ensure we have at least one embedding dimension
            if document_embeddings.shape[1] == 0:
                pooled_embeddings.append(document_embeddings)
                cluster_assignments.append([])
                continue
            
            # Determine the number of clusters based on pool_factor
            num_clusters = max(num_embeddings // pool_factor, 1)
            
            # If we have fewer embeddings than clusters, just use all embeddings
            if num_clusters >= num_embeddings:
                pooled_embeddings.append(document_embeddings)
                cluster_assignments.append(list(range(1, num_embeddings + 1)))
                continue
            
            # Normalize embeddings for spherical k-means (cosine similarity)
            # Spherical k-means works on unit vectors
            embeddings_normalized = torch.nn.functional.normalize(
                embeddings_to_pool, p=2, dim=1
            )
            
            # Convert to numpy for fastkmeans (it expects numpy arrays)
            embeddings_np = embeddings_normalized.cpu().float().numpy()
            embedding_dim = embeddings_np.shape[1]
            
            # Initialize and train fastkmeans
            # TODO: Using GPU causes some weird issues I haven't been able to debug. 
            # Given that we're only throwing ~300 (max doclen) embeddings in any given clustering, it's not worth the hassle to figure out why.
            kmeans = fastkmeans.FastKMeans(
                embedding_dim,
                num_clusters,
                niter=10,  # Number of iterations
                gpu=False,
                verbose=False,
                seed=42,
            )
            
            # Train the kmeans model
            kmeans.train(embeddings_np)
            
            # Get cluster assignments for each embedding
            cluster_labels_np = kmeans.predict(embeddings_np)
            
            # Convert cluster labels to torch tensor for indexing
            cluster_labels_tensor = torch.from_numpy(cluster_labels_np).to(device=device)
            
            # Convert cluster labels to list (for storage)
            cluster_labels_list = cluster_labels_np.tolist()
            
            # Store cluster assignments (convert to 1-indexed to match hierarchical)
            # fastkmeans uses 0-indexed, but we need 1-indexed to match hierarchical format
            cluster_assignments.append([label + 1 for label in cluster_labels_list])
            
            # Pool embeddings within each cluster (optionally weighted)
            pooled_document_embeddings = []
            for cluster_id in range(num_clusters):
                # Find indices of embeddings belonging to this cluster
                cluster_indices = torch.where(cluster_labels_tensor == cluster_id)[0]
                if cluster_indices.numel() > 0:
                    cluster_emb = embeddings_to_pool[cluster_indices]
                    if doc_weights is not None:
                        # Weighted average: extract weights for this cluster
                        cluster_weights = doc_weights[actual_protected:][cluster_indices]
                        # Normalize weights to sum to 1 (with safety check for zero sum)
                        weight_sum = cluster_weights.sum()
                        if weight_sum > 0:
                            cluster_weights = cluster_weights / weight_sum
                            # Weighted average
                            cluster_embedding = (cluster_emb * cluster_weights.unsqueeze(1)).sum(dim=0)
                        else:
                            # Fallback to unweighted average if all weights are zero
                            cluster_embedding = cluster_emb.mean(dim=0)
                    else:
                        # Unweighted average
                        cluster_embedding = cluster_emb.mean(dim=0)
                    pooled_document_embeddings.append(cluster_embedding)
            
            # Re-append protected embeddings
            pooled_document_embeddings.extend(protected_embeddings)
            pooled_embeddings.append(torch.stack(tensors=pooled_document_embeddings))
        
        return pooled_embeddings, cluster_assignments
    
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply pooling compression to embeddings and update artifacts to maintain 1:1 mapping.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Shape-matched artifacts will be updated to match
            the pooled embeddings. Metadata artifacts are passed through unchanged.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (pooled_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with pooled embeddings.
            Metadata artifacts are passed through unchanged.
        """
        # Skip pooling if pool_factor is 1 (no pooling)
        if self.config.pool_factor == 1:
            return embeddings, artifacts
        
        # Extract or compute weights if weighted pooling is enabled
        weights = None
        if self.config.weight_by is not None:
            if self.config.weight_by == "attention":
                if "attention_scores" not in artifacts:
                    raise ValueError(
                        "PoolingStrategy requires 'attention_scores' artifact when weight_by='attention'. "
                        "Ensure attention_scores are provided when encoding."
                    )
                weights = artifacts["attention_scores"]
            elif self.config.weight_by == "leverage":
                if "leverage_scores" not in artifacts:
                    raise ValueError(
                        "PoolingStrategy requires 'leverage_scores' artifact when weight_by='leverage'. "
                        "Ensure leverage_scores are provided when encoding."
                    )
                weights = artifacts["leverage_scores"]
            elif self.config.weight_by in ["idf", "tfidf"]:
                if "input_ids" not in artifacts:
                    raise ValueError(
                        f"PoolingStrategy requires 'input_ids' artifact when weight_by='{self.config.weight_by}'. "
                        "Ensure input_ids are provided when encoding."
                    )
                input_ids = artifacts["input_ids"]
                
                # Get or compute TF-IDF stats
                if "tfidf_stats" in artifacts:
                    stats = artifacts["tfidf_stats"]
                else:
                    # Compute TF-IDF stats from input_ids
                    from .utils import TokenTFIDFStats
                    tokenized_docs = [doc_input_ids.cpu().tolist() for doc_input_ids in input_ids]
                    stats = TokenTFIDFStats(num_docs=len(tokenized_docs))
                    stats.fit(tokenized_docs, show_progress=False)
                
                # Compute scores for each document
                weights = []
                for doc_idx, doc_input_ids in enumerate(input_ids):
                    doc_tokens = doc_input_ids.cpu().tolist()
                    doc_scores = []
                    for token_id in doc_tokens:
                        if self.config.weight_by == "idf":
                            score = stats.get_idf(token_id)
                        else:  # tfidf
                            score = stats.get_tfidf(doc_idx, token_id)
                        doc_scores.append(score)
                    # Convert to tensor on same device as embeddings
                    weights.append(torch.tensor(doc_scores, device=embeddings[doc_idx].device, dtype=embeddings[doc_idx].dtype))
        
        # Apply pooling based on clustering method
        if self.config.clustering_method == "hierarchical":
            pooled_embeddings, cluster_assignments = self._pool_embeddings_hierarchical(
                documents_embeddings=embeddings,
                pool_factor=self.config.pool_factor,
                protected_tokens=self.config.protected_tokens,
                weights=weights,
            )
        elif self.config.clustering_method == "spherical":
            pooled_embeddings, cluster_assignments = self._pool_embeddings_spherical(
                documents_embeddings=embeddings,
                pool_factor=self.config.pool_factor,
                protected_tokens=self.config.protected_tokens,
                weights=weights,
            )
        elif self.config.clustering_method == "window":
            pooled_embeddings, cluster_assignments = self._pool_embeddings_window(
                documents_embeddings=embeddings,
                pool_factor=self.config.pool_factor,
                protected_tokens=self.config.protected_tokens,
                weights=weights,
                stride=self.config.stride,
            )
        elif self.config.clustering_method == "random":
            pooled_embeddings, cluster_assignments = self._pool_embeddings_random(
                documents_embeddings=embeddings,
                pool_factor=self.config.pool_factor,
                protected_tokens=self.config.protected_tokens,
                weights=weights,
            )
        else:
            raise ValueError(
                f"Unknown clustering method: {self.config.clustering_method}. "
                f"Must be 'hierarchical', 'spherical', 'window', or 'random'."
            )
        
        # Update shape-matched artifacts to match pooled embeddings
        # Use cluster assignments to map original tokens to pooled tokens
        updated_artifacts = {}
        for artifact_name, artifact_value in artifacts.items():
            if isinstance(artifact_value, list):
                # Shape-matched artifact: need to pool it using cluster assignments
                pooled_artifacts = []
                for doc_idx, artifact_tokens in enumerate(artifact_value):
                    doc_embeddings = embeddings[doc_idx]
                    pooled_doc_embeddings = pooled_embeddings[doc_idx]
                    
                    # Protected tokens map to themselves
                    protected_artifact_tokens = artifact_tokens[:self.config.protected_tokens]
                    
                    # Convert artifact_tokens to list if it's a tensor
                    if isinstance(artifact_tokens, torch.Tensor):
                        artifact_tokens_list = artifact_tokens.tolist()
                    else:
                        artifact_tokens_list = list(artifact_tokens)
                    
                    # Map pooled tokens using cluster assignments
                    pooled_artifact_tokens_list = list(protected_artifact_tokens)
                    
                    # Use cluster assignments to select representative tokens
                    doc_cluster_labels = cluster_assignments[doc_idx]
                    if doc_cluster_labels:
                        # Find the maximum cluster ID to determine number of clusters
                        max_cluster_id = max(doc_cluster_labels)
                        num_clusters = max_cluster_id
                    else:
                        num_clusters = 0
                    
                    # For each cluster, select the first token as representative
                    for cluster_id in range(1, num_clusters + 1):
                        cluster_token_indices = [
                            i for i, label in enumerate(doc_cluster_labels)
                            if label == cluster_id
                        ]
                        if cluster_token_indices:
                            # Use the first token in the cluster as representative
                            original_idx = cluster_token_indices[0] + self.config.protected_tokens
                            if original_idx < len(artifact_tokens_list):
                                pooled_artifact_tokens_list.append(artifact_tokens_list[original_idx])
                    
                    # Ensure we have the right number of tokens (should match pooled embeddings)
                    while len(pooled_artifact_tokens_list) < len(pooled_doc_embeddings):
                        # Pad with last token if needed (shouldn't happen, but safety check)
                        if artifact_tokens_list:
                            pooled_artifact_tokens_list.append(artifact_tokens_list[-1])
                        else:
                            break
                    
                    pooled_artifact_tokens_list = pooled_artifact_tokens_list[:len(pooled_doc_embeddings)]
                    
                    # Convert back to tensor if original was tensor
                    if isinstance(artifact_value[doc_idx], torch.Tensor):
                        pooled_artifacts.append(torch.tensor(pooled_artifact_tokens_list, device=artifact_value[doc_idx].device, dtype=artifact_value[doc_idx].dtype))
                    else:
                        pooled_artifacts.append(pooled_artifact_tokens_list)
                
                updated_artifacts[artifact_name] = pooled_artifacts
            else:
                # Metadata artifact: pass through unchanged
                updated_artifacts[artifact_name] = artifact_value
        
        return pooled_embeddings, updated_artifacts


@dataclass
class CompressionConfig:
    """High level configuration for compression strategies.
    
    Composes multiple compression strategies that are applied in sequence.
    Strategies are directly serializable via their serialize() and from_dict() methods.
    """
    
    strategies: list[CompressionStrategy] = field(default_factory=list)
    description: str = ""  # Optional natural language description for logging/display
    
    def create_compressor(self) -> "Compressor":
        """
        Create a Compressor instance from this config.
        
        Returns
        -------
        Compressor
            Runtime compressor that executes strategies
        """
        return Compressor(self.strategies)
    
    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "description": self.description,
            "strategies": [strategy.serialize() for strategy in self.strategies],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CompressionConfig":
        """
        Deserialize a CompressionConfig from a dictionary.
        
        Parameters
        ----------
        data
            Dictionary containing serialized config data
        
        Returns
        -------
        CompressionConfig
            Deserialized configuration
        """
        strategies = []
        # Map strategy types to their classes
        strategy_classes = {
            "idf_pruning": IDFPruningStrategy,
            "attention_pruning": AttentionPruningStrategy,
            "compactor_pruning": CompactorPruningStrategy,
            "pooling": PoolingStrategy,
        }
        
        for strategy_data in data.get("strategies", []):
            strategy_type = strategy_data.get("type")
            if strategy_type not in strategy_classes:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            strategy_cls = strategy_classes[strategy_type]
            strategies.append(strategy_cls.from_dict(strategy_data))
        
        return cls(
            strategies=strategies,
            description=data.get("description", ""),
        )


class Compressor:
    """
    Runtime object that executes compression strategies in sequence.
    
    Maintains 1:1 mapping between embeddings and artifacts throughout compression pipeline.
    """
    
    def __init__(self, strategies: list[CompressionStrategy]):
        self.strategies = strategies
        self._validate_strategies()
    
    def _validate_strategies(self) -> None:
        """Validate that strategies are properly configured."""
        for strategy in self.strategies:
            if not isinstance(strategy, CompressionStrategy):
                raise TypeError(
                    f"Strategy {strategy} is not an instance of CompressionStrategy"
                )
    
    def get_required_artifacts(self) -> set[str]:
        """
        Get set of all required artifacts across all strategies.
        
        Returns
        -------
        set[str]
            Set of artifact keys required by any strategy
        """
        required = set()
        for strategy in self.strategies:
            required.update(strategy.required_artifacts)
        return required
    
    def get_required_artifacts_with_args(self) -> dict[str, dict[str, Any]]:
        """
        Get dictionary of all required artifacts with their hook creation arguments.
        
        This method aggregates artifact requirements from all strategies, collecting
        the hook creation arguments needed for each artifact.
        
        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping artifact names to their hook creation arguments.
            For example:
            - "attention_scores" -> {"head_reduction": "sum"}
            - "leverage_scores" -> {"sketch_dim": 64, "head_reduction": "sum"}
            - "input_ids" -> {} (no arguments needed)
        
        Notes
        -----
        If multiple strategies require the same artifact with different arguments,
        the arguments from the last strategy requiring it will be used.
        Each strategy specifies its requirements via get_artifact_requirements().
        """
        artifacts_with_args: dict[str, dict[str, Any]] = {}
        
        for strategy in self.strategies:
            strategy_requirements = strategy.get_artifact_requirements()
            for artifact_name, args in strategy_requirements.items():
                # Store the args (will overwrite if multiple strategies need same artifact)
                artifacts_with_args[artifact_name] = args
        
        return artifacts_with_args
    
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply all compression strategies in sequence.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Can contain:
            - Shape-matched artifacts: `list[torch.Tensor]` (one per document)
            - Metadata artifacts: `Any` (corpus-level or document-level metadata)
            Must contain all required artifacts.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (compressed_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with compressed embeddings.
            Metadata artifacts are passed through unchanged.
        
        Raises
        ------
        ValueError
            If required artifacts are missing
        """
        # Validate required artifacts are present
        required = self.get_required_artifacts()
        missing = required - set(artifacts.keys())
        if missing:
            raise ValueError(
                f"Missing required artifacts: {missing}. "
                f"Required by strategies: {required}"
            )
        
        # Copy artifacts to avoid modifying input
        # Shape-matched artifacts: deep copy tensors
        # Metadata artifacts: shallow copy (they're typically immutable or shared)
        copied_artifacts = {}
        for k, v in artifacts.items():
            if isinstance(v, list):
                copied_artifacts[k] = [a.clone() for a in v]
            else:
                copied_artifacts[k] = v  # Metadata: pass through
        
        result_embeddings = [emb.clone() for emb in embeddings]
        
        # Apply each strategy in sequence
        for strategy in self.strategies:
            result_embeddings, copied_artifacts = strategy.compress(result_embeddings, copied_artifacts)
            
            # Validate 1:1 mapping maintained for shape-matched artifacts only
            num_embeddings = len(result_embeddings)
            for artifact_name, artifact_value in copied_artifacts.items():
                if isinstance(artifact_value, list):
                    if len(artifact_value) != num_embeddings:
                        raise ValueError(
                            f"Strategy {strategy.name} broke 1:1 mapping: "
                            f"{num_embeddings} embeddings but {len(artifact_value)} {artifact_name} artifacts"
                        )
        
        return result_embeddings, copied_artifacts
    
    def compress_parallel(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
        batch_size: int = 100,
        num_workers: Optional[int] = None,
        show_progress: bool = False,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply all compression strategies in sequence using parallel processing.
        
        This method applies each strategy in sequence, but each strategy processes
        documents in parallel batches for improved performance.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Can contain:
            - Shape-matched artifacts: `list[torch.Tensor]` (one per document)
            - Metadata artifacts: `Any` (corpus-level or document-level metadata)
            Must contain all required artifacts.
        batch_size
            Number of documents to process in each batch. Defaults to 100.
        num_workers
            Number of worker threads to use per strategy. If None, defaults to min(batch_size, number of documents).
        show_progress
            If True, shows a progress bar during parallel compression. Defaults to False.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (compressed_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with compressed embeddings.
            Metadata artifacts are passed through unchanged.
        
        Raises
        ------
        ValueError
            If required artifacts are missing
        """
        # Validate required artifacts are present
        required = self.get_required_artifacts()
        missing = required - set(artifacts.keys())
        if missing:
            raise ValueError(
                f"Missing required artifacts: {missing}. "
                f"Required by strategies: {required}"
            )
        
        # Copy artifacts to avoid modifying input
        # Shape-matched artifacts: deep copy tensors
        # Metadata artifacts: shallow copy (they're typically immutable or shared)
        copied_artifacts = {}
        for k, v in artifacts.items():
            if isinstance(v, list):
                copied_artifacts[k] = [a.clone() for a in v]
            else:
                copied_artifacts[k] = v  # Metadata: pass through
        
        result_embeddings = [emb.clone() for emb in embeddings]
        
        # Apply each strategy in sequence, using parallel processing for each
        for strategy in self.strategies:
            result_embeddings, copied_artifacts = strategy.compress_parallel(
                result_embeddings,
                copied_artifacts,
                batch_size=batch_size,
                num_workers=num_workers,
                show_progress=show_progress,
            )
            
            # Validate 1:1 mapping maintained for shape-matched artifacts only
            num_embeddings = len(result_embeddings)
            for artifact_name, artifact_value in copied_artifacts.items():
                if isinstance(artifact_value, list):
                    if len(artifact_value) != num_embeddings:
                        raise ValueError(
                            f"Strategy {strategy.name} broke 1:1 mapping: "
                            f"{num_embeddings} embeddings but {len(artifact_value)} {artifact_name} artifacts"
                        )
        
        return result_embeddings, copied_artifacts

# Hooks

def make_attention_score_hook(results_list: list[torch.Tensor], head_reduction: Literal["sum", "max"] = "sum") -> Callable[[nn.Module, tuple, tuple], None]:
    def attention_score_hook(module: nn.Module, input: tuple, output: tuple) -> None:
        """
        Compute per-token importance scores by aggregating attention across all heads
        and source tokens. For each token j, the importance score is the sum of attention paid
        to token j over all heads and all source tokens.
        Places results in results_list.
        """
        # Try to extract attention scores from output
        attention_scores = None

        hidden_states = input[0]
        attention_mask = input[1] if len(input) > 1 else None
        
        # Compute attention scores manually for ModernBertAttention
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get num_heads from config
        num_heads = module.num_heads
        head_dim = hidden_size // num_heads
        
        # Get Q, K, V from Wqkv projection
        qkv = module.Wqkv(hidden_states)  # (batch, seq_len, 3*hidden_size)
        qkv = qkv.view(batch_size, seq_len, 3, hidden_size)  # (batch, seq_len, 3, hidden_size)
        q, k, v = qkv.chunk(3, dim=2)  # Each: (batch, seq_len, 1, hidden_size)
        q = q.squeeze(2)  # (batch, seq_len, hidden_size)
        k = k.squeeze(2)  # (batch, seq_len, hidden_size)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        # -> (batch, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            elif attention_mask.dim() == 3:
                mask = attention_mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            else:
                mask = attention_mask
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Aggregate attention scores to per-token importance scores
        # attention_scores: (batch, num_heads, seq_len, seq_len)
        # Sum over source tokens (dim=1) for per-token importance (batch, num_heads,seq_len)
        per_head_importance_scores = attention_scores.sum(dim=1) # (batch, num_heads, seq_len)
        match head_reduction:
            case "sum":
                importance_scores = per_head_importance_scores.sum(dim=1) # (batch, seq_len)
            case "max":
                importance_scores = per_head_importance_scores.max(dim=1).values # (batch, seq_len)
            case _:
                raise ValueError(f"Invalid head reduction: {head_reduction}")
        results_list.append(importance_scores.detach().clone())
        
    return attention_score_hook

def _compute_leverage_scores_right_sketch(k_tensor: torch.Tensor, sketch_dim: Optional[int] = None, lambda_reg: float = 1e-3) -> torch.Tensor:
    """
    Gemini's implementation of Compactor's right-sketch leverage score computation (replaces SVD with Cholesky decomp)
    
    Args:
        k_tensor: (batch, num_heads, seq_len, head_dim) 
        sketch_dim: int
        lambda_reg: regularization parameter
    """
    B, H, N, D = k_tensor.shape
    
    # shared sketch matrix for all heads
    
    if sketch_dim is None:
        sketch_dim = D
        k_hat = k_tensor
    else:
        # (B, H, N, D) @ (D, k) -> (B, H, N, k)
        phi = torch.randn((D, sketch_dim), device=k_tensor.device, dtype=k_tensor.dtype) * (sketch_dim ** -0.5)
        k_hat = k_tensor @ phi
    
    # Compute Gram Matrix G = K_hat^T * K_hat
    # (B, H, k, N) @ (B, H, N, k) -> (B, H, k, k)
    G = k_hat.transpose(-1, -2) @ k_hat
    
    # Regularization (Ridge)
    eye = torch.eye(sketch_dim, device=k_tensor.device, dtype=k_tensor.dtype)
    G = G + lambda_reg * eye
    
    # Compute whitened vectors U = K_hat * G^(-1/2)
    # Instead of explicit SVD (slow on batch), we can use Cholesky or Inverse since D is small (head dim).
    # We want row_norms(K_hat @ G^-0.5)^2 = diag(K_hat @ G^-1 @ K_hat^T)
    # Efficiently: sum((K_hat @ G^-1) * K_hat, dim=-1)
    
    G_inv = torch.linalg.inv(G) # (B, H, k, k)
    
    # Project K_hat by inverse covariance
    k_whitened = torch.matmul(k_hat, G_inv) # (B, H, N, k)
    
    # Dot product with self to get squared Mahalanobis distance
    scores = (k_whitened * k_hat).sum(dim=-1) # (B, H, N)
    
    return scores

def make_leverage_score_hook(results_list: list[torch.Tensor], sketch_dim: Optional[int] = None, head_reduction: Literal["sum", "max"] = "sum") -> Callable[[nn.Module, tuple, tuple], None]:
    def leverage_score_hook(module: nn.Module, input: tuple, output: tuple) -> None:
        """
        Inspired by Compactor (http://arxiv.org/abs/2507.08143)
        Compute per-token leverage scores on K. If sketch_dim is provided, sketch down to sketch_dim to make it more efficient.
        Scores are the sum of leverage scores for a token over all heads
        """
        hidden_states = input[0]
        attention_mask = input[1] if len(input) > 1 else None
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_heads = module.num_heads
        head_dim = hidden_size // num_heads
        
        qkv = module.Wqkv(hidden_states)  # (batch, seq_len, 3*hidden_size)
        qkv = qkv.view(batch_size, seq_len, 3, hidden_size)  # (batch, seq_len, 3, hidden_size)
        q, k, v = qkv.chunk(3, dim=2)  # Each: (batch, seq_len, 1, hidden_size)
        k = k.squeeze(2)  # (batch, seq_len, hidden_size)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        leverage_scores = _compute_leverage_scores_right_sketch(k, sketch_dim, lambda_reg=1e-3) # (batch, num_heads, seq_len)

        match head_reduction: # (batch, seq_len)
            case "sum":
                leverage_scores = leverage_scores.sum(dim=1)
            case "max":
                leverage_scores = leverage_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid head reduction: {head_reduction}. Only 'sum' and 'max' are supported.")

        results_list.append(leverage_scores.detach().clone())
    return leverage_score_hook