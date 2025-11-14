from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional
from abc import ABC, abstractmethod
import torch

from .utils import TokenTFIDFStats


@dataclass
class IDFPruningConfig:
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
    apply_to_queries
        Whether to apply pruning to queries. Defaults to False (documents only).
    protected_tokens
        Number of leading tokens to always retain (CLS / prefixes).
    stats
        Required :class:`TokenTFIDFStats` object containing precomputed IDF scores.
    use_tfidf
        If True, uses TF-IDF scoring (considers in-document frequency).
        If False, uses only IDF scoring. Defaults to False.
    track_pruned_tokens
        If True, tracks which tokens were pruned from each document. Access via
        ``strategy.get_pruned_tokens()`` after encoding. Defaults to False.
    ignore_tokens
        Optional set of token IDs to exclude from pruning. These tokens will never be
        pruned regardless of their IDF scores. Useful for special tokens (pad, sep, etc.)
        or domain-specific tokens that should always be preserved. Defaults to None.
    """

    mode: Literal["global", "document"] = "document"
    top_k: Optional[int] = None
    threshold: Optional[float] = None
    apply_to_queries: bool = False
    protected_tokens: int = 1
    stats: TokenTFIDFStats = None  # Required, but can't enforce in dataclass default
    use_tfidf: bool = False
    track_pruned_tokens: bool = False
    ignore_tokens: Optional[set[int]] = None

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
        if self.stats is None:
            raise ValueError(
                "IDF pruning requires `stats` to be provided. Use TokenTFIDFStats.fit() to compute statistics."
            )


@dataclass
class PoolingConfig:
    pool_factor: int = 1
    protected_tokens: int = 1
    clustering_method: Literal["hierarchical", "spherical"] = "hierarchical"


@dataclass
class CompressionConfig:
    """High level configuration for pruning and pooling."""

    pruning: list[IDFPruningConfig] = field(default_factory=list)
    pooling: list[PoolingConfig] = field(default_factory=list)
    description: str = ""  # Optional natural language description for logging/display


class CompressionContext:
    """
    Runtime container used to execute pruning/pooling strategies inside ``encode``.
    """

    def __init__(self, config: Optional[CompressionConfig] = None) -> None:
        config = config or CompressionConfig()
        self.config = config
        self.pruning_strategies: list[PruningStrategy] = [
            IDFPruningStrategy(cfg) for cfg in config.pruning
        ]

    def has_pruning(self) -> bool:
        return bool(self.pruning_strategies)

    def apply_pruning(
        self,
        *,
        token_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        base_mask: torch.Tensor,
        is_query: bool,
    ) -> torch.Tensor:
        """
        Compute the final mask after running all pruning strategies.

        Parameters
        ----------
        token_embeddings
            Per-document token embeddings tensor.
        input_ids
            Per-document token IDs tensor (needed for token-based strategies like IDF).
        base_mask
            Base mask that already excludes padding and skiplist tokens.
        is_query
            Whether this is a query (some strategies only apply to documents).

        Returns
        -------
        torch.Tensor
            Final pruning mask (boolean tensor).
        """
        if not self.pruning_strategies:
            return base_mask

        composed_mask = base_mask.clone()
        for strategy in self.pruning_strategies:
            composed_mask = strategy.update_mask(
                token_embeddings=token_embeddings,
                input_ids=input_ids,
                current_mask=composed_mask,
                is_query=is_query,
            )
        return composed_mask

    def finalize(self) -> dict[str, TokenTFIDFStats]:
        """
        Finalize any strategy that collected statistics.
        Returns a mapping of strategy identifiers to generated stats.
        """
        artifacts: dict[str, TokenTFIDFStats] = {}
        for strategy in self.pruning_strategies:
            stats = strategy.finalize()
            if stats is not None:
                artifacts[strategy.name] = stats
        return artifacts

    def get_pruned_tokens(
        self, strategy_name: str = "idf"
    ) -> Optional[list[dict[int, tuple[int, float]]]]:
        """
        Get pruned tokens from a specific pruning strategy.

        Parameters
        ----------
        strategy_name : str
            Name of the pruning strategy (default: "idf").

        Returns
        -------
        list[dict[int, tuple[int, float]]] or None
            List of dictionaries (one per document) mapping token_id to (position, idf_score).
            Returns None if strategy not found or tracking was disabled.
        """
        for strategy in self.pruning_strategies:
            if strategy.name == strategy_name:
                return strategy.get_pruned_tokens()
        return None


class PruningStrategy(ABC):
    """
    Base interface for pruning strategies.
    """

    name: str = "pruning"

    @abstractmethod
    def update_mask(
        self,
        *,
        token_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        current_mask: torch.Tensor,
        is_query: bool,
    ) -> torch.Tensor:
        """
        Update the pruning mask based on this strategy.

        Parameters
        ----------
        token_embeddings
            Per-document token embeddings tensor.
        input_ids
            Per-document token IDs tensor.
        current_mask
            Current mask (already excludes padding/skiplist tokens).
        is_query
            Whether this is a query.

        Returns
        -------
        torch.Tensor
            Updated mask (boolean tensor).
        """
        pass

    @abstractmethod
    def finalize(self) -> Optional[TokenTFIDFStats]:
        pass


class IDFPruningStrategy(PruningStrategy):
    name = "idf"

    def __init__(self, config: IDFPruningConfig) -> None:
        self.config = config
        # Track pruned tokens per document if requested
        # List of dicts: [{token_id: (position, idf_score), ...}, ...]
        self._pruned_tokens: list[dict[int, tuple[int, float]]] = (
            [] if config.track_pruned_tokens else None
        )
        self._batch_offset = 0  # Track absolute document index across batches

    def update_mask(
        self,
        *,
        token_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        current_mask: torch.Tensor,
        is_query: bool,
    ) -> torch.Tensor:
        if is_query and not self.config.apply_to_queries:
            return current_mask

        if self.config.mode == "global":
            return self._prune_global(
                input_ids=input_ids,
                current_mask=current_mask,
                stats=self.config.stats,
            )
        else:  # "document" mode
            return self._prune_per_document(
                input_ids=input_ids,
                current_mask=current_mask,
                stats=self.config.stats,
            )

    def _prune_global(
        self,
        *,
        input_ids: torch.Tensor,
        current_mask: torch.Tensor,
        stats: TokenTFIDFStats,
    ) -> torch.Tensor:
        """
        Prune k token types globally across the entire batch.

        Identifies the k unique token types with the **lowest IDF scores** (i.e., highest
        document frequency, most common across documents, least informative), then removes
        **all occurrences** of those k token types from all documents in the batch.
        Different documents may end up with different numbers of tokens removed depending
        on how many times those token types appear in each document.
        """
        batch_size = input_ids.shape[0]
        composed_mask = current_mask.clone()

        # Initialize tracking for this batch if enabled
        if self._pruned_tokens is not None:
            batch_pruned = [{} for _ in range(batch_size)]

        # Step 1: Collect all unique token types and their IDF scores, along with their positions
        token_positions = {}  # token_id -> list of (doc_idx, token_idx)
        token_scores = {}  # token_id -> IDF score

        for doc_idx in range(batch_size):
            doc_mask = current_mask[doc_idx]
            active_indices = torch.nonzero(doc_mask, as_tuple=False).squeeze(-1)

            if active_indices.numel() <= self.config.protected_tokens:
                continue

            protected = min(self.config.protected_tokens, active_indices.numel())
            candidate_indices = active_indices[protected:]

            if candidate_indices.numel() == 0:
                continue

            tokens = input_ids[doc_idx, candidate_indices].tolist()

            # Filter out ignored tokens if specified
            if self.config.ignore_tokens:
                filtered_indices = []
                filtered_tokens = []
                for idx, token_id in enumerate(tokens):
                    if token_id not in self.config.ignore_tokens:
                        filtered_indices.append(idx)
                        filtered_tokens.append(token_id)

                if not filtered_tokens:
                    continue

                candidate_indices = candidate_indices[filtered_indices]
                tokens = filtered_tokens

            # Get IDF scores for these tokens (only need to compute once per unique token)
            unique_tokens_in_doc = list(set(tokens))
            scores = self._compute_scores(
                tokens=unique_tokens_in_doc,
                stats=stats,
                doc_tokens=tokens if self.config.use_tfidf else None,
            )
            token_to_score = dict(zip(unique_tokens_in_doc, scores))

            # Record positions and scores
            for idx, token_id in enumerate(tokens):
                token_idx = candidate_indices[idx].item()

                if token_id not in token_positions:
                    token_positions[token_id] = []
                    token_scores[token_id] = token_to_score[token_id].item()

                token_positions[token_id].append((doc_idx, token_idx))

        if not token_positions:
            if self._pruned_tokens is not None:
                self._pruned_tokens.extend(batch_pruned)
                self._batch_offset += batch_size
            return composed_mask

        # Step 2: Select which token types to prune based on top_k or threshold
        if self.config.top_k is not None:
            # Sort token types by IDF score (ascending) and take bottom-k token types
            # Lower IDF = higher document frequency = more common = less informative
            sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1])
            k = min(self.config.top_k, len(sorted_tokens))
            tokens_to_prune = {token_id for token_id, score in sorted_tokens[:k]}
        else:
            # Prune token types with IDF below threshold (common tokens)
            threshold = self.config.threshold or float("-inf")
            tokens_to_prune = {
                token_id
                for token_id, score in token_scores.items()
                if score < threshold
            }

        # Step 3: Prune all occurrences of the selected token types
        for token_id in tokens_to_prune:
            score = token_scores[token_id]
            for doc_idx, token_idx in token_positions[token_id]:
                composed_mask[doc_idx, token_idx] = False
                if self._pruned_tokens is not None:
                    batch_pruned[doc_idx][token_id] = (token_idx, score)

        if self._pruned_tokens is not None:
            self._pruned_tokens.extend(batch_pruned)
            self._batch_offset += batch_size

        return composed_mask

    def _prune_per_document(
        self,
        *,
        input_ids: torch.Tensor,
        current_mask: torch.Tensor,
        stats: TokenTFIDFStats,
    ) -> torch.Tensor:
        """
        Prune k tokens per document independently.

        Removes the k tokens with the **lowest IDF scores** from each document (i.e., most common
        across documents, least informative). Each document has the same number of tokens removed (up to k).
        """
        batch_size = input_ids.shape[0]
        composed_mask = current_mask.clone()

        # Initialize tracking for this batch if enabled
        if self._pruned_tokens is not None:
            batch_pruned = [{} for _ in range(batch_size)]

        for doc_idx in range(batch_size):
            doc_mask = current_mask[doc_idx]
            active_indices = torch.nonzero(doc_mask, as_tuple=False).squeeze(-1)

            if active_indices.numel() <= self.config.protected_tokens:
                continue

            protected = min(self.config.protected_tokens, active_indices.numel())
            protected_indices = active_indices[:protected]
            candidate_indices = active_indices[protected:]

            if candidate_indices.numel() == 0:
                continue

            tokens = input_ids[doc_idx, candidate_indices].tolist()

            # Filter out ignored tokens if specified
            if self.config.ignore_tokens:
                filtered_indices = []
                filtered_tokens = []
                for idx, token_id in enumerate(tokens):
                    if token_id not in self.config.ignore_tokens:
                        filtered_indices.append(idx)
                        filtered_tokens.append(token_id)

                if not filtered_tokens:
                    continue

                candidate_indices = candidate_indices[filtered_indices]
                tokens = filtered_tokens

            scores = self._compute_scores(
                tokens=tokens,
                stats=stats,
                doc_tokens=tokens if self.config.use_tfidf else None,
            )

            if self.config.top_k is not None:
                # Prune k tokens with lowest IDF scores (most common across documents)
                k = min(self.config.top_k, candidate_indices.numel())
                if k > 0:
                    # Get indices of k lowest IDF scores
                    # largest=False returns smallest values (lowest IDF = highest doc freq)
                    bottomk = torch.topk(scores, k=k, largest=False).indices
                    prune_candidates = candidate_indices[bottomk]
                    prune_scores = scores[bottomk]
                else:
                    prune_candidates = torch.tensor([], dtype=torch.long)
                    prune_scores = torch.tensor([], dtype=torch.float32)
            else:
                # Prune tokens with IDF below threshold (common tokens)
                threshold = self.config.threshold or float("-inf")
                prune_mask = scores < threshold
                prune_candidates = candidate_indices[prune_mask]
                prune_scores = scores[prune_mask]

            # Update mask and track pruned tokens
            for idx, (token_position, score) in enumerate(
                zip(prune_candidates.tolist(), prune_scores.tolist())
            ):
                composed_mask[doc_idx, token_position] = False
                if self._pruned_tokens is not None:
                    token_id = input_ids[doc_idx, token_position].item()
                    batch_pruned[doc_idx][token_id] = (token_position, score)

        if self._pruned_tokens is not None:
            self._pruned_tokens.extend(batch_pruned)
            self._batch_offset += batch_size

        return composed_mask

    def _compute_scores(
        self,
        *,
        tokens: list[int],
        stats: TokenTFIDFStats,
        doc_tokens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Compute IDF or TF-IDF scores for tokens.

        Lower scores indicate more common tokens (high document frequency, low IDF).
        These low-scoring tokens are the ones that will be pruned.
        """
        device = torch.device("cpu")
        length = len(tokens)
        scores = torch.zeros(length, dtype=torch.float32, device=device)
        if length == 0:
            return scores

        counts = None
        if self.config.use_tfidf and doc_tokens is not None:
            counts = Counter(doc_tokens)

        for idx, token_id in enumerate(tokens):
            idf = stats.get_idf(token_id)
            if counts is not None:
                # TF-IDF: combines term frequency and inverse document frequency
                tf = counts[token_id] / len(doc_tokens)
                scores[idx] = tf * idf
            else:
                # Pure IDF: lower values = more common across documents
                scores[idx] = idf
        return scores

    def finalize(self) -> Optional[TokenTFIDFStats]:
        """Return the stats object if tracking was enabled."""
        return self.config.stats

    def get_pruned_tokens(self) -> Optional[list[dict[int, tuple[int, float]]]]:
        """
        Get the tracked pruned tokens if tracking was enabled.

        Returns
        -------
        list[dict[int, tuple[int, float]]] or None
            List of dictionaries (one per document) mapping token_id to (position, idf_score).
            Returns None if track_pruned_tokens was False.

        Examples
        --------
        >>> pruned = strategy.get_pruned_tokens()
        >>> if pruned:
        ...     for doc_idx, doc_pruned in enumerate(pruned):
        ...         print(f"Doc {doc_idx}: pruned {len(doc_pruned)} tokens")
        ...         for token_id, (position, score) in doc_pruned.items():
        ...             print(f"  Token {token_id} at position {position} (IDF={score:.3f})")
        """
        return self._pruned_tokens

    def reset_tracking(self) -> None:
        """Reset the pruned tokens tracking. Useful when processing multiple corpora."""
        if self._pruned_tokens is not None:
            self._pruned_tokens = []
            self._batch_offset = 0
