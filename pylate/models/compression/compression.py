from .base import *
from .idf_pruning import IDFPruningStrategy
from .attention_pruning import AttentionPruningStrategy
from .leverage_score_pruning import LeverageScorePruningStrategy
from .importance_pruning import ImportancePruningStrategy
from .importance_pooling import ImportancePoolingStrategy
from .hybrid_importance_pooling import (
    HybridPoolingConfig,
    HybridImportanceClusteringPoolingStrategy,
)
from .random_pruning import RandomPruningConfig, RandomPruningStrategy
from .random_pooling import RandomPoolingConfig, RandomPoolingStrategy
from .pooling import PoolingStrategy


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
            "leverage_score_pruning": LeverageScorePruningStrategy,
            "importance_pruning": ImportancePruningStrategy,
            "importance_pooling": ImportancePoolingStrategy,
            # legacy name for hybrid importance + clustering pooling
            "hybrid_importance_clustering_pooling": HybridImportanceClusteringPoolingStrategy,
            # new name is dynamic (hybrid_imp+clust_pooling_*); handled below
            "random_pruning": RandomPruningStrategy,
            "random_pooling": RandomPoolingStrategy,
            "pooling": PoolingStrategy,
        }
        
        for strategy_data in data.get("strategies", []):
            strategy_type = strategy_data.get("type")
            # Handle dynamic hybrid naming (hybrid_imp+clust_pooling_*)
            if strategy_type and strategy_type.startswith("hybrid_imp+clust_pooling"):
                strategy_cls = HybridImportanceClusteringPoolingStrategy
            else:
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
    
    
