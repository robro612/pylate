from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Literal, Optional
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

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "apply_to_queries": self.apply_to_queries,
            "protected_tokens": self.protected_tokens,
            "use_tfidf": self.use_tfidf,
        }


@dataclass
class PoolingConfig:
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


@dataclass
class CompressionConfig:
    """High level configuration for pruning and pooling."""

    pruning: list[IDFPruningConfig] = field(default_factory=list)
    pooling: list[PoolingConfig] = field(default_factory=list)
    description: str = ""  # Optional natural language description for logging/display

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
            "pruning": [p.serialize() for p in self.pruning],
            "pooling": [p.serialize() for p in self.pooling],
        }


@dataclass
class CompressionExperimentConfig:
    """
    Configuration for running multiple compression experiments in a single encoding pass.

    This enables efficient exploration of compression trade-offs by running the expensive
    transformer forward pass once, then applying multiple cheap compression strategies.

    For large datasets (e.g., MS-MARCO), use ``storage_mode="disk"`` to stream results to disk.
    For small datasets that fit in memory, use ``storage_mode="memory"`` or ``storage_mode="cpu"``.

    Parameters
    ----------
    configs : List[CompressionConfig | CompressionContext | None]
        List of compression configurations to apply. Use None for baseline (no compression).
        Can mix :class:`CompressionConfig` (will create contexts internally) and
        :class:`CompressionContext` (for tracking pruned tokens).
    output_dir : Path | str
        Directory to save compressed embeddings and metadata. Each config gets its own file.
        Created automatically if it doesn't exist (unless it exists and ``overwrite=False``).
    config_names : Optional[List[str]]
        Names for each config (used for filenames and metadata). If None, uses
        ``config.description`` or auto-generates names like "config_0", "config_1", etc.
        Names are sanitized (spaces → underscores, slashes removed) for safe filenames.
    storage_mode : Literal["disk", "memory", "cpu"]
        How to store results:

        - ``"disk"``: Stream to disk immediately after each batch (lowest memory, scales to any
          dataset size). Recommended for large datasets (>100K documents).
        - ``"cpu"``: Keep in CPU memory (medium memory, faster than disk). Frees GPU memory
          after each config but keeps results in RAM. Good for medium datasets.
        - ``"memory"``: Keep in GPU memory (highest memory, fastest but limited by GPU RAM).
          Only suitable for small datasets (<10K documents).

        Default: ``"disk"``
    save_batch_size : Optional[int]
        How many documents to accumulate before writing to disk (batching reduces I/O overhead).
        Only used if ``storage_mode="disk"``. If None, saves after each encoding batch.
        Larger values reduce disk writes but increase memory usage.
        Default: None (immediate save)
    overwrite : bool
        Whether to overwrite existing output directory. If False and directory exists,
        raises ValueError. Default: False
    compression : Literal["none", "gzip", "lz4"]
        PyTorch save compression format. Options:

        - ``"none"``: No compression (fastest save/load, largest files)
        - ``"gzip"``: Gzip compression (3x smaller files, slower save/load)
        - ``"lz4"``: LZ4 compression (2x smaller files, moderate speed)

        Default: ``"none"``
    run_id : str
        Run ID for consistent file naming and tracking. Shard files use format:
        ``run-{run_id}.config-{config_idx}.pt`` (matching runfile format).
        The run_id is saved in experiment_metadata.json for tracking and matching shards to configs.

    Examples
    --------
    >>> from datetime import datetime
    >>> from pylate.models import CompressionExperimentConfig, CompressionConfig, IDFPruningConfig
    >>>
    >>> # Define multiple compression strategies
    >>> experiment = CompressionExperimentConfig(
    ...     configs=[
    ...         None,  # Baseline
    ...         CompressionConfig(description="IDF k=10", pruning=[IDFPruningConfig(...)]),
    ...         CompressionConfig(description="IDF k=20", pruning=[IDFPruningConfig(...)]),
    ...     ],
    ...     output_dir="experiments/msmarco_compression",
    ...     storage_mode="disk",  # Stream to disk for large dataset
    ...     run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),  # Required for tracking
    ... )
    >>>
    >>> # Run experiment (encode once, apply all configs)
    >>> results = model.encode(documents, compression_config=experiment)
    >>>
    >>> # Load and evaluate each config
    >>> for i, config in enumerate(experiment.configs):
    ...     embeddings = results.load_embeddings(i)
    ...     # ... create index, evaluate ...
    """

    configs: list[CompressionConfig | CompressionContext | None]
    output_dir: Path | str
    run_id: str
    config_names: Optional[list[str]] = None
    storage_mode: Literal["disk", "memory", "cpu"] = "disk"
    save_batch_size: Optional[int] = None
    overwrite: bool = False
    compression: Literal["none", "gzip", "lz4"] = "none"

    def __post_init__(self) -> None:
        # Validate
        if not self.configs:
            raise ValueError("configs must be a non-empty list")

        self.output_dir = Path(self.output_dir)

        # Check for existing directory
        if self.output_dir.exists() and not self.overwrite:
            raise ValueError(
                f"Output directory {self.output_dir} already exists. "
                "Set overwrite=True or choose a different directory."
            )

        # Generate config names if not provided
        if self.config_names is None:
            self.config_names = []
            for i, cfg in enumerate(self.configs):
                if cfg is None:
                    name = "baseline"
                elif isinstance(cfg, CompressionContext):
                    name = (
                        cfg.config.description
                        if cfg.config.description
                        else f"config_{i}"
                    )
                else:  # CompressionConfig
                    name = cfg.description if cfg.description else f"config_{i}"
                self.config_names.append(name)

        # Validate config_names length
        if len(self.config_names) != len(self.configs):
            raise ValueError(
                f"config_names length ({len(self.config_names)}) must match "
                f"configs length ({len(self.configs)})"
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_file(self, config_idx: int) -> Path:
        """
        Get the output file path for a specific config.

        Parameters
        ----------
        config_idx : int
            Index of the config (0-based)

        Returns
        -------
        Path
            Path to the embeddings file for this config
        """
        return self.output_dir / f"run-{self.run_id}.config-{config_idx}.pt"

    def serialize(self) -> dict:
        """
        Serialize this configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        serialized_configs = []
        for cfg in self.configs:
            if cfg is None:
                serialized_configs.append({"type": "none", "description": "No compression"})
            elif isinstance(cfg, CompressionContext):
                serialized_configs.append(cfg.serialize())
            elif isinstance(cfg, CompressionConfig):
                serialized_configs.append(cfg.serialize())
            else:
                # Fallback for unexpected types
                serialized_configs.append({"type": "unknown", "description": str(cfg)})
        
        return {
            "configs": serialized_configs,
            "config_names": self.config_names,
            "output_dir": str(self.output_dir),
            "storage_mode": self.storage_mode,
            "save_batch_size": self.save_batch_size,
            "overwrite": self.overwrite,
            "compression": self.compression,
            "run_id": self.run_id,
        }


@dataclass
class CompressionExperimentResults:
    """
    Results from a compression experiment with multiple configs.

    Returned by :meth:`ColBERT.encode` when passed a :class:`CompressionExperimentConfig`.
    Provides convenient access to embeddings and metadata for each compression configuration.

    Attributes
    ----------
    experiment_config : CompressionExperimentConfig
        The experiment configuration used
    output_files : Optional[List[Path]]
        Paths to saved embedding files (if ``storage_mode="disk"``), None otherwise
    embeddings : Optional[List[List[torch.Tensor]]]
        In-memory embeddings (if ``storage_mode="memory"`` or ``"cpu"``), None otherwise.
        Structure: ``embeddings[config_idx][doc_idx]`` = embedding tensor for that document
    contexts : List[CompressionContext | None]
        Compression contexts for each config. Useful for accessing tracked data like
        pruned tokens via :meth:`get_pruned_tokens`.
    statistics : Dict[str, Any]
        Summary statistics including:

        - ``num_documents``: Total documents encoded
        - ``num_configs``: Number of compression configs tested
        - ``config_token_counts``: List of total tokens per config
        - ``avg_tokens_per_doc``: List of average tokens per document per config
        - ``encoding_time``: Time spent in transformer forward pass (seconds)
        - ``compression_times``: List of time spent in compression per config (seconds)
        - ``total_time``: Total time including encoding and compression (seconds)

    Examples
    --------
    >>> # Run experiment
    >>> results = model.encode(documents, compression_config=experiment)
    >>>
    >>> # Access statistics
    >>> print(f"Encoded {results.statistics['num_documents']} documents")
    >>> print(f"Baseline: {results.statistics['avg_tokens_per_doc'][0]:.1f} tokens/doc")
    >>> print(f"Compressed: {results.statistics['avg_tokens_per_doc'][1]:.1f} tokens/doc")
    >>>
    >>> # Load embeddings for a specific config
    >>> embeddings = results.load_embeddings(config_idx=1, device="cuda")
    >>>
    >>> # Iterate over all configs
    >>> for config, embeddings in results:
    ...     # ... evaluate this config ...
    >>>
    >>> # Access pruned tokens (if tracking enabled)
    >>> pruned = results.get_pruned_tokens(config_idx=1, strategy_name="idf")
    """

    experiment_config: CompressionExperimentConfig
    output_files: Optional[list[Path]] = None
    embeddings: Optional[list[list[torch.Tensor]]] = None
    contexts: list[CompressionContext | None] = None
    statistics: dict[str, Any] = field(default_factory=dict)

    def load_embeddings(
        self, config_idx: int, device: str = "cpu"
    ) -> list[torch.Tensor]:
        """
        Load embeddings for a specific config.

        Parameters
        ----------
        config_idx : int
            Index of the config to load (0-based)
        device : str
            Device to load tensors to (e.g., "cpu", "cuda", "cuda:0")

        Returns
        -------
        list[torch.Tensor]
            List of embedding tensors (one per document)

        Raises
        ------
        ValueError
            If config_idx is out of range or embeddings not available
        IndexError
            If config_idx is out of bounds
        """
        if config_idx < 0 or config_idx >= len(self.experiment_config.configs):
            raise IndexError(
                f"config_idx {config_idx} out of range "
                f"[0, {len(self.experiment_config.configs)})"
            )

        if self.experiment_config.storage_mode in ["memory", "cpu"]:
            if self.embeddings is None:
                raise ValueError("Embeddings not available in memory")
            return [emb.to(device) for emb in self.embeddings[config_idx]]
        else:  # disk
            if self.output_files is None:
                raise ValueError("Output files not available")
            data = torch.load(self.output_files[config_idx], map_location=device)
            return data["embeddings"]

    def get_pruned_tokens(
        self, config_idx: int, strategy_name: str = "idf"
    ) -> Optional[list[dict[int, tuple[int, float]]]]:
        """
        Get pruned tokens for a specific config (if tracking was enabled).

        Parameters
        ----------
        config_idx : int
            Index of the config
        strategy_name : str
            Name of the pruning strategy (default: "idf")

        Returns
        -------
        list[dict[int, tuple[int, float]]] or None
            List of dictionaries (one per document) mapping token_id to (position, idf_score).
            Returns None if tracking was not enabled or strategy not found.

        Examples
        --------
        >>> pruned = results.get_pruned_tokens(config_idx=1)
        >>> if pruned:
        ...     for doc_idx, doc_pruned in enumerate(pruned):
        ...         print(f"Doc {doc_idx}: pruned {len(doc_pruned)} tokens")
        """
        if self.contexts and self.contexts[config_idx]:
            return self.contexts[config_idx].get_pruned_tokens(strategy_name)
        return None

    def __iter__(self):
        """
        Iterate over (config, embeddings) pairs.

        Yields
        ------
        tuple[CompressionConfig | CompressionContext | None, list[torch.Tensor]]
            Each iteration yields (config, embeddings) for one configuration

        Examples
        --------
        >>> for config, embeddings in results:
        ...     description = config.description if config else "Baseline"
        ...     print(f"{description}: {len(embeddings)} documents")
        """
        for i in range(len(self.experiment_config.configs)):
            yield self.experiment_config.configs[i], self.load_embeddings(i)

    def save_summary(self, output_file: Optional[Path] = None) -> None:
        """
        Save a human-readable summary of the experiment results.

        Parameters
        ----------
        output_file : Path, optional
            Where to save the summary. If None, saves to
            ``{output_dir}/experiment_summary.txt``
        """
        if output_file is None:
            output_file = self.experiment_config.output_dir / "experiment_summary.txt"

        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPRESSION EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Output Directory: {self.experiment_config.output_dir}\n")
            f.write(f"Storage Mode: {self.experiment_config.storage_mode}\n")
            f.write(
                f"Number of Documents: {self.statistics.get('num_documents', 'N/A')}\n"
            )
            f.write(f"Number of Configs: {len(self.experiment_config.configs)}\n\n")

            f.write("Timing:\n")
            f.write(
                f"  Encoding Time: {self.statistics.get('encoding_time', 0):.2f}s\n"
            )
            f.write(f"  Total Time: {self.statistics.get('total_time', 0):.2f}s\n\n")

            f.write("=" * 80 + "\n")
            f.write("CONFIGURATIONS\n")
            f.write("=" * 80 + "\n\n")

            for i, config in enumerate(self.experiment_config.configs):
                name = self.experiment_config.config_names[i]
                f.write(f"[{i}] {name}\n")
                f.write("-" * 80 + "\n")

                token_count = self.statistics.get(
                    "config_token_counts", [None] * (i + 1)
                )[i]
                avg_tokens = self.statistics.get(
                    "avg_tokens_per_doc", [None] * (i + 1)
                )[i]
                comp_time = self.statistics.get("compression_times", [None] * (i + 1))[
                    i
                ]

                f.write(
                    f"  Total Tokens: {token_count:,}\n"
                    if token_count is not None
                    else "  Total Tokens: N/A\n"
                )
                f.write(
                    f"  Avg Tokens/Doc: {avg_tokens:.2f}\n"
                    if avg_tokens is not None
                    else "  Avg Tokens/Doc: N/A\n"
                )
                f.write(
                    f"  Compression Time: {comp_time:.2f}s\n"
                    if comp_time is not None
                    else "  Compression Time: N/A\n"
                )

                if i > 0 and token_count is not None:
                    baseline_count = self.statistics.get("config_token_counts", [None])[
                        0
                    ]
                    if baseline_count is not None:
                        reduction_pct = (1 - token_count / baseline_count) * 100
                        f.write(f"  Reduction vs Baseline: {reduction_pct:.2f}%\n")

                f.write("\n")

    def serialize(self) -> dict:
        """
        Serialize this results object to a JSON-compatible dictionary.

        Note: Embeddings are not included in the serialization as they are large tensors.
        Use load_embeddings() to access embeddings separately.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return {
            "experiment_config": self.experiment_config.serialize(),
            "output_files": [str(f) for f in self.output_files] if self.output_files else None,
            "statistics": self.statistics,
            # Note: embeddings and contexts are not serialized as they contain tensors/objects
            # that are not JSON-serializable
        }


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

    def serialize(self) -> dict:
        """
        Serialize this context's configuration to a JSON-compatible dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary representation
        """
        return self.config.serialize()

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

        # Step 1: Get tokens to prune from corpus stats (IDF-based, not TF-IDF for global pruning)
        if self.config.top_k is not None:
            # Greedily get k tokens, filtering out ignored tokens
            # Get all tokens sorted by IDF (lowest first), filter out ignored tokens, then take k
            ignore_set = self.config.ignore_tokens or set()
            all_low_idf_tokens = stats.get_low_idf_tokens(k=min(self.config.top_k + len(ignore_set), len(stats.idf_scores)))
            # Filter out ignored tokens and take first k
            filtered_tokens = [token_id for token_id in all_low_idf_tokens if token_id not in ignore_set]
            tokens_to_prune = set(filtered_tokens[:self.config.top_k])
        else:
            threshold = self.config.threshold or float("-inf")
            tokens_to_prune = set(stats.get_low_idf_tokens(threshold=threshold))
            # Filter out ignored tokens if specified
            if self.config.ignore_tokens:
                tokens_to_prune -= self.config.ignore_tokens

        if not tokens_to_prune:
            if self._pruned_tokens is not None:
                self._pruned_tokens.extend([{} for _ in range(batch_size)])
                self._batch_offset += batch_size
            return composed_mask

        # Step 2: Create a tensor mask for tokens to prune
        # Convert tokens_to_prune to a tensor for efficient comparison
        tokens_to_prune_tensor = torch.tensor(
            list(tokens_to_prune), dtype=input_ids.dtype, device=input_ids.device
        )

        # Step 3: Apply pruning mask elementwise
        # For each position: prune if token_id is in tokens_to_prune AND not protected
        # Initialize tracking if enabled
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

            # Get token IDs for candidate positions
            candidate_token_ids = input_ids[doc_idx, candidate_indices]

            # Check which tokens are in tokens_to_prune using torch.isin (more efficient)
            is_prunable = torch.isin(candidate_token_ids, tokens_to_prune_tensor)

            # Update mask: set to False for prunable tokens
            prune_indices = candidate_indices[is_prunable]
            composed_mask[doc_idx, prune_indices] = False

            # Track pruned tokens if enabled
            if self._pruned_tokens is not None:
                for token_idx in prune_indices:
                    token_id = input_ids[doc_idx, token_idx].item()
                    idf_score = stats.get_idf(token_id)
                    batch_pruned[doc_idx][token_id] = (token_idx.item(), idf_score)

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
