from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union, Collection
from abc import ABC, abstractmethod
import torch
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
class PoolingConfig(CompressionStrategyConfigBase):
    pool_factor: int = 1
    protected_tokens: int = 1
    clustering_method: Literal["hierarchical", "spherical"] = "hierarchical"
    show_progress_bar: bool = False

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
            "show_progress_bar": self.show_progress_bar,
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
    
    required_artifacts: list[str] = ["input_ids"]  # Requires input_ids to compute TF-IDF stats
    
    def __init__(self, config: IDFPruningConfig):
        """
        Initialize the IDF pruning strategy.
        
        Parameters
        ----------
        config
            IDF pruning configuration specifying mode, top_k/threshold, protected_tokens, etc.
        """
        self.config = config
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
        input_ids: list[torch.Tensor],
    ) -> set[int]:
        """
        Identify token types to prune globally based on lowest scores.
        
        Parameters
        ----------
        stats
            TF-IDF statistics
        input_ids
            List of input_id tensors
        
        Returns
        -------
        set[int]
            Set of token IDs to prune globally
        """
        # Collect all unique token types and their scores
        token_scores: dict[int, float] = {}
        
        for doc_idx, doc_input_ids in enumerate(input_ids):
            doc_tokens = doc_input_ids.cpu().tolist()
            
            # Get unique tokens in this document
            unique_tokens = set(doc_tokens)
            
            for token_id in unique_tokens:
                # Skip protected tokens and ignored tokens
                if token_id in doc_tokens[:self.config.protected_tokens]:
                    continue
                if self.config.ignore_token_ids and token_id in self.config.ignore_token_ids:
                    continue
                
                # Use minimum score across documents (most conservative)
                score = self._get_token_score(stats, doc_idx, token_id)
                if token_id not in token_scores or score < token_scores[token_id]:
                    token_scores[token_id] = score
        
        # Select tokens to prune based on top_k or threshold
        if self.config.top_k is not None:
            # Sort by score (ascending) and take top_k lowest
            # Note: ignored tokens are already excluded from token_scores, so we'll prune exactly top_k non-ignored tokens
            sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1])
            # Take min(top_k, len(sorted_tokens)) to handle cases where we have fewer candidates than top_k
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
        if self.config.track_pruned_tokens:
            self._pruned_tokens = []
        else:
            self._pruned_tokens = None
        
        # Compute TF-IDF statistics from input_ids
        stats = self._compute_tfidf_stats(input_ids)
        
        # Determine tokens to prune based on mode
        # Store keep masks for artifact updates
        keep_masks: list[list[bool]] = []
        
        if self.config.mode == "global":
            # Global mode: identify token types to prune globally
            tokens_to_prune_global = self._get_tokens_to_prune_global(stats, input_ids)
            
            # Prune all occurrences of these token types from all documents
            pruned_embeddings = []
            updated_input_ids = []
            
            criterion_str = f"top_k={self.config.top_k}" if self.config.top_k else f"threshold={self.config.threshold}"
            iterator = tqdm(
                enumerate(zip(embeddings, input_ids)),
                desc=f"IDF pruning (global, {criterion_str})",
                total=len(embeddings),
                disable=not self.config.show_progress_bar,
            )
            
            for doc_idx, (doc_embeddings, doc_input_ids) in iterator:
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
                    self._pruned_tokens.append(pruned_token_ids)
                
                # Store keep mask for artifact updates
                keep_masks.append(keep_mask)
        
        else:  # document mode
            # Document mode: prune independently per document
            pruned_embeddings = []
            updated_input_ids = []
            
            criterion_str = f"top_k={self.config.top_k}" if self.config.top_k else f"threshold={self.config.threshold}"
            iterator = tqdm(
                enumerate(zip(embeddings, input_ids)),
                desc=f"IDF pruning (document, {criterion_str})",
                total=len(embeddings),
                disable=not self.config.show_progress_bar,
            )
            
            for doc_idx, (doc_embeddings, doc_input_ids) in iterator:
                # Get positions to prune for this document
                positions_to_prune = self._get_tokens_to_prune_document(
                    stats, doc_idx, doc_input_ids
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
                # Metadata artifact: pass through unchanged
                updated_artifacts[artifact_name] = artifact_value
        
        return pruned_embeddings, updated_artifacts


class PoolingStrategy(CompressionStrategy):
    """
    Pooling strategy that wraps the hierarchical/spherical pooling logic from ColBERT.
    
    This strategy pools embeddings by clustering similar token embeddings together
    and averaging them, reducing the number of tokens per document while preserving
    semantic information.
    
    The hierarchical method uses Ward's linkage clustering on cosine similarity distances.
    The spherical method uses fastkmeans clustering on the embeddings.
    """
    
    required_artifacts: list[str] = []  # Pooling doesn't require any artifacts
    
    def __init__(self, config: PoolingConfig):
        """
        Initialize the pooling strategy.
        
        Parameters
        ----------
        config
            Pooling configuration specifying pool_factor, protected_tokens, and clustering_method
        """
        if config.pool_factor <= 0:
            raise ValueError("`pool_factor` must be a positive integer.")
        if config.protected_tokens < 0:
            raise ValueError("`protected_tokens` must be non-negative.")
        
        self.config = config
    
    @property
    def name(self) -> str:
        """Name of this compression strategy."""
        return f"pooling-{self.config.clustering_method}_k-{self.config.pool_factor}_p-{self.config.protected_tokens}"
    
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
        )
        return cls(config)
    
    def _pool_embeddings_hierarchical(
        self,
        documents_embeddings: list[torch.Tensor],
        pool_factor: int,
        protected_tokens: int,
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """
        Pools the embeddings hierarchically by clustering and averaging them.
        
        This method wraps the exact same logic as ColBERT.pool_embeddings_hierarchical.
        
        Parameters
        ----------
        documents_embeddings
            A list of embeddings for each document.
        pool_factor
            Factor to determine the number of clusters.
        protected_tokens
            Number of tokens to protect from pooling at the start of each document.
        
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
            documents_embeddings,
            desc=f"Hierarchical pooling (factor={pool_factor})",
            disable=not self.config.show_progress_bar,
        )
        
        for document_embeddings in iterator:
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
            
            # Pool embeddings within each cluster
            pooled_document_embeddings = []
            for cluster_id in range(1, num_clusters + 1):
                cluster_indices = torch.where(
                    condition=torch.tensor(
                        data=cluster_labels == cluster_id, device=device
                    )
                )[0]
                if cluster_indices.numel() > 0:
                    cluster_embedding = embeddings_to_pool[cluster_indices].mean(dim=0)
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
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """
        Pools the embeddings using spherical clustering via fastkmeans.
        
        This method uses fastkmeans to perform k-means clustering on the embeddings,
        then averages embeddings within each cluster to create pooled representations.
        
        Parameters
        ----------
        documents_embeddings
            A list of embeddings for each document.
        pool_factor
            Factor to determine the number of clusters.
        protected_tokens
            Number of tokens to protect from pooling at the start of each document.
        
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
            documents_embeddings,
            desc=f"Spherical pooling (factor={pool_factor})",
            disable=not self.config.show_progress_bar,
        )
        
        for document_embeddings in iterator:
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
            use_gpu = device.type == "cuda"
            kmeans = fastkmeans.FastKMeans(
                embedding_dim,
                num_clusters,
                niter=10,  # Number of iterations
                gpu=use_gpu,
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
            
            # Pool embeddings within each cluster by averaging
            pooled_document_embeddings = []
            for cluster_id in range(num_clusters):
                # Find indices of embeddings belonging to this cluster
                cluster_indices = torch.where(cluster_labels_tensor == cluster_id)[0]
                if cluster_indices.numel() > 0:
                    # Average the embeddings in this cluster
                    cluster_embedding = embeddings_to_pool[cluster_indices].mean(dim=0)
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
        
        # Apply pooling based on clustering method
        if self.config.clustering_method == "hierarchical":
            pooled_embeddings, cluster_assignments = self._pool_embeddings_hierarchical(
                documents_embeddings=embeddings,
                pool_factor=self.config.pool_factor,
                protected_tokens=self.config.protected_tokens,
            )
        elif self.config.clustering_method == "spherical":
            pooled_embeddings, cluster_assignments = self._pool_embeddings_spherical(
                documents_embeddings=embeddings,
                pool_factor=self.config.pool_factor,
                protected_tokens=self.config.protected_tokens,
            )
        else:
            raise ValueError(
                f"Unknown clustering method: {self.config.clustering_method}. "
                f"Must be 'hierarchical' or 'spherical'."
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
                    num_clusters = max(len(doc_cluster_labels) // self.config.pool_factor, 1)
                    
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