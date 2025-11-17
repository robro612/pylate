from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union, Collection
from abc import ABC, abstractmethod
import torch

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
    ignore_token_ids
        Optional set of token IDs to exclude from pruning. These tokens will never be
        pruned regardless of their IDF scores. Useful for special tokens (pad, sep, etc.)
        or domain-specific tokens that should always be preserved. Defaults to None.
    """

    mode: Literal["global", "document"] = "document"
    top_k: Optional[int] = None
    threshold: Optional[float] = None
    protected_tokens: int = 1
    use_tfidf: bool = False
    track_pruned_tokens: bool = False
    ignore_token_ids: Optional[Collection[int]] = None

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
        }
        # Convert set to list for JSON serialization
        if self.ignore_token_ids is not None:
            result["ignore_token_ids"] = list(self.ignore_token_ids)
        return result
    
    @property
    def strategy_type(self) -> str:
        return "idf_pruning"


@dataclass
class PoolingConfig(CompressionStrategyConfigBase):
    pool_factor: int = 1
    protected_tokens: int = 1
    clustering_method: Literal["hierarchical", "spherical"] = "hierarchical"

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "pool_factor": self.pool_factor,
            "protected_tokens": self.protected_tokens,
            "clustering_method": self.clustering_method,
        }
    
    @property
    def strategy_type(self) -> str:
        return "pooling"


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
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this compression strategy."""
        pass
    
    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """Strategy type identifier for serialization (e.g., 'idf_pruning', 'pooling')."""
        pass
    
    @abstractmethod
    def serialize(self) -> dict:
        """Serialize this strategy to a JSON-compatible dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "CompressionStrategy":
        """Create a strategy instance from a serialized dictionary."""
        pass
    
    @abstractmethod
    def compress(
        self,
        embeddings: list[torch.Tensor],
        artifacts: CompressionArtifacts,
    ) -> tuple[list[torch.Tensor], CompressionArtifacts]:
        """
        Apply compression to embeddings and update artifacts to maintain 1:1 mapping.
        
        Parameters
        ----------
        embeddings
            List of embedding tensors (one per document)
        artifacts
            Dictionary of artifacts. Can contain:
            - Shape-matched artifacts: `list[torch.Tensor]` (one per document, must match embedding shape)
            - Metadata artifacts: `Any` (corpus-level or document-level metadata)
            Will be copied internally, so modifications are safe.
        
        Returns
        -------
        tuple[list[torch.Tensor], CompressionArtifacts]
            (compressed_embeddings, updated_artifacts)
            Shape-matched artifacts maintain 1:1 mapping with compressed embeddings.
            Metadata artifacts are passed through unchanged.
        """
        pass


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