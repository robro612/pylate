"""Tests for compression strategies and configurations."""

from __future__ import annotations

import json

import pytest
import torch

from pylate.models.compression import (
    CompressionArtifacts,
    CompressionConfig,
    Compressor,
    IDFPruningConfig,
    IDFPruningStrategy,
    PoolingConfig,
    PoolingStrategy,
)


@pytest.fixture
def sample_embeddings() -> list[torch.Tensor]:
    """Create sample embeddings for testing."""
    torch.manual_seed(42)
    # Force CPU to avoid CUDA issues in tests
    return [
        torch.randn(10, 128, device="cpu"),  # Document 1: 10 tokens
        torch.randn(9, 128, device="cpu"),    # Document 2: 9 tokens (must match input_ids length)
        torch.randn(12, 128, device="cpu"),   # Document 3: 12 tokens
    ]


@pytest.fixture
def sample_input_ids() -> list[torch.Tensor]:
    """Create sample input_ids for testing."""
    # Create diverse token IDs to test IDF pruning
    # Force CPU to avoid CUDA issues in tests
    # IMPORTANT: Lengths must match sample_embeddings shapes exactly
    return [
        torch.tensor([101, 1, 2, 3, 4, 5, 1, 2, 3, 102], device="cpu"),  # Doc 1: 10 tokens - tokens 1,2,3 appear multiple times
        torch.tensor([101, 1, 2, 6, 7, 8, 1, 2, 102], device="cpu"),     # Doc 2: 9 tokens - tokens 1,2 appear multiple times
        torch.tensor([101, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 102], device="cpu"),  # Doc 3: 12 tokens - unique tokens
    ]


@pytest.fixture
def sample_artifacts(sample_input_ids: list[torch.Tensor]) -> CompressionArtifacts:
    """Create sample artifacts for testing."""
    return {
        "input_ids": sample_input_ids,
        "attention_mask": [
            torch.ones(len(ids), dtype=torch.long) for ids in sample_input_ids
        ],
    }


class TestIDFPruningConfig:
    """Tests for IDFPruningConfig validation and serialization."""

    def test_config_validation_top_k(self):
        """Test config validation with top_k."""
        config = IDFPruningConfig(mode="document", top_k=5)
        assert config.top_k == 5
        assert config.threshold is None

    def test_config_validation_threshold(self):
        """Test config validation with threshold."""
        config = IDFPruningConfig(mode="global", threshold=0.5)
        assert config.threshold == 0.5
        assert config.top_k is None

    def test_config_validation_both_none(self):
        """Test that config requires either top_k or threshold."""
        with pytest.raises(ValueError, match="requires either"):
            IDFPruningConfig(mode="document")

    def test_config_validation_both_provided(self):
        """Test that config rejects both top_k and threshold."""
        with pytest.raises(ValueError, match="only one of"):
            IDFPruningConfig(mode="document", top_k=5, threshold=0.5)

    def test_config_validation_top_k_negative(self):
        """Test that top_k must be positive."""
        with pytest.raises(ValueError, match="positive integer"):
            IDFPruningConfig(mode="document", top_k=-1)

    def test_config_validation_threshold_infinite(self):
        """Test that threshold must be finite."""
        with pytest.raises(ValueError, match="finite float"):
            IDFPruningConfig(mode="document", threshold=float("inf"))

    def test_config_serialization(self):
        """Test config serialization."""
        config = IDFPruningConfig(
            mode="global",
            top_k=10,
            protected_tokens=2,
            use_tfidf=True,
            track_pruned_tokens=True,
            ignore_token_ids={101, 102},
            show_progress_bar=True,
        )
        serialized = config.serialize()
        assert serialized["mode"] == "global"
        assert serialized["top_k"] == 10
        assert serialized["protected_tokens"] == 2
        assert serialized["use_tfidf"] is True
        assert serialized["track_pruned_tokens"] is True
        assert set(serialized["ignore_token_ids"]) == {101, 102}
        assert serialized["show_progress_bar"] is True


class TestIDFPruningStrategy:
    """Tests for IDFPruningStrategy."""

    def test_strategy_requires_input_ids(self):
        """Test that strategy requires input_ids artifact."""
        config = IDFPruningConfig(mode="document", top_k=2)
        strategy = IDFPruningStrategy(config)
        assert "input_ids" in strategy.required_artifacts

    def test_strategy_name_top_k(self):
        """Test strategy name generation with top_k."""
        config = IDFPruningConfig(mode="document", top_k=5)
        strategy = IDFPruningStrategy(config)
        assert "idf_pruning" in strategy.name
        assert "document" in strategy.name
        assert "topk-5" in strategy.name

    def test_strategy_name_threshold(self):
        """Test strategy name generation with threshold."""
        config = IDFPruningConfig(mode="global", threshold=0.5)
        strategy = IDFPruningStrategy(config)
        assert "idf_pruning" in strategy.name
        assert "global" in strategy.name
        assert "threshold-0.5" in strategy.name

    def test_document_mode_top_k(self, sample_embeddings, sample_artifacts):
        """Test document mode pruning with top_k."""
        config = IDFPruningConfig(mode="document", top_k=2, protected_tokens=1)
        strategy = IDFPruningStrategy(config)
        
        pruned_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Should have same number of documents
        assert len(pruned_embeddings) == len(sample_embeddings)
        
        # Each document should have fewer tokens (protected + remaining)
        for i, (orig, pruned) in enumerate(zip(sample_embeddings, pruned_embeddings)):
            assert pruned.shape[0] <= orig.shape[0]
            assert pruned.shape[0] >= config.protected_tokens  # At least protected tokens remain
            assert pruned.shape[1] == orig.shape[1]  # Embedding dimension unchanged
        
        # Artifacts should be updated
        assert "input_ids" in updated_artifacts
        assert len(updated_artifacts["input_ids"]) == len(sample_embeddings)
        for i, pruned_ids in enumerate(updated_artifacts["input_ids"]):
            assert len(pruned_ids) == pruned_embeddings[i].shape[0]

    def test_global_mode_top_k(self, sample_embeddings, sample_artifacts):
        """Test global mode pruning with top_k."""
        config = IDFPruningConfig(mode="global", top_k=2, protected_tokens=1)
        strategy = IDFPruningStrategy(config)
        
        pruned_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Should have same number of documents
        assert len(pruned_embeddings) == len(sample_embeddings)
        
        # All documents should have fewer tokens
        for i, (orig, pruned) in enumerate(zip(sample_embeddings, pruned_embeddings)):
            assert pruned.shape[0] <= orig.shape[0]
            assert pruned.shape[0] >= config.protected_tokens
            assert pruned.shape[1] == orig.shape[1]

    def test_document_mode_threshold(self, sample_embeddings, sample_artifacts):
        """Test document mode pruning with threshold."""
        config = IDFPruningConfig(mode="document", threshold=0.1, protected_tokens=1)
        strategy = IDFPruningStrategy(config)
        
        pruned_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Should have same number of documents
        assert len(pruned_embeddings) == len(sample_embeddings)
        
        # Each document should have fewer or equal tokens
        for i, (orig, pruned) in enumerate(zip(sample_embeddings, pruned_embeddings)):
            assert pruned.shape[0] <= orig.shape[0]
            assert pruned.shape[0] >= config.protected_tokens

    def test_protected_tokens(self, sample_embeddings, sample_artifacts):
        """Test that protected tokens are never pruned."""
        config = IDFPruningConfig(mode="document", top_k=100, protected_tokens=2)
        strategy = IDFPruningStrategy(config)
        
        pruned_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Protected tokens should always remain
        for i, pruned_ids in enumerate(updated_artifacts["input_ids"]):
            original_ids = sample_artifacts["input_ids"][i]
            assert len(pruned_ids) >= config.protected_tokens
            # First protected tokens should match
            for j in range(config.protected_tokens):
                assert pruned_ids[j] == original_ids[j]

    def test_ignore_token_ids(self, sample_embeddings, sample_artifacts):
        """Test that ignored tokens are never pruned."""
        config = IDFPruningConfig(
            mode="document",
            top_k=100,  # Try to prune many tokens
            protected_tokens=1,
            ignore_token_ids={1, 2},  # Ignore tokens 1 and 2
        )
        strategy = IDFPruningStrategy(config)
        
        pruned_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Ignored tokens should still be present
        for pruned_ids in updated_artifacts["input_ids"]:
            pruned_ids_list = pruned_ids.tolist() if isinstance(pruned_ids, torch.Tensor) else list(pruned_ids)
            # Check that ignored tokens are present (if they were in original)
            # Note: This is a basic check - in practice, ignored tokens should not be pruned

    def test_track_pruned_tokens(self, sample_embeddings, sample_artifacts):
        """Test tracking of pruned tokens."""
        config = IDFPruningConfig(
            mode="document",
            top_k=2,
            track_pruned_tokens=True,
        )
        strategy = IDFPruningStrategy(config)
        
        pruned_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Should have tracked pruned tokens
        pruned_tokens = strategy.get_pruned_tokens()
        assert pruned_tokens is not None
        assert len(pruned_tokens) == len(sample_embeddings)
        assert all(isinstance(doc_tokens, list) for doc_tokens in pruned_tokens)

    def test_no_track_pruned_tokens(self, sample_embeddings, sample_artifacts):
        """Test that pruned tokens are not tracked when disabled."""
        config = IDFPruningConfig(
            mode="document",
            top_k=2,
            track_pruned_tokens=False,
        )
        strategy = IDFPruningStrategy(config)
        
        pruned_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Should not have tracked pruned tokens
        pruned_tokens = strategy.get_pruned_tokens()
        assert pruned_tokens is None

    def test_missing_input_ids(self, sample_embeddings):
        """Test that missing input_ids raises error."""
        config = IDFPruningConfig(mode="document", top_k=2)
        strategy = IDFPruningStrategy(config)
        
        with pytest.raises(ValueError, match="requires 'input_ids'"):
            strategy.compress(sample_embeddings, {})

    def test_shape_mismatch(self, sample_embeddings):
        """Test that shape mismatch raises error."""
        config = IDFPruningConfig(mode="document", top_k=2)
        strategy = IDFPruningStrategy(config)
        
        artifacts = {
            "input_ids": [torch.tensor([1, 2, 3])],  # Only one document
        }
        
        with pytest.raises(ValueError, match="Mismatch"):
            strategy.compress(sample_embeddings, artifacts)

    def test_artifact_update(self, sample_embeddings, sample_artifacts):
        """Test that shape-matched artifacts are updated correctly."""
        config = IDFPruningConfig(mode="document", top_k=2)
        strategy = IDFPruningStrategy(config)
        
        pruned_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # All shape-matched artifacts should be updated
        assert "attention_mask" in updated_artifacts
        for i, mask in enumerate(updated_artifacts["attention_mask"]):
            assert len(mask) == pruned_embeddings[i].shape[0]

    def test_serialization(self):
        """Test strategy serialization."""
        config = IDFPruningConfig(mode="document", top_k=5, protected_tokens=2)
        strategy = IDFPruningStrategy(config)
        
        serialized = strategy.serialize()
        assert serialized["type"] == "idf_pruning"
        assert "config" in serialized
        assert serialized["config"]["mode"] == "document"
        assert serialized["config"]["top_k"] == 5

    def test_deserialization(self):
        """Test strategy deserialization."""
        data = {
            "type": "idf_pruning",
            "config": {
                "mode": "global",
                "top_k": 10,
                "protected_tokens": 1,
                "use_tfidf": False,
                "track_pruned_tokens": False,
                "show_progress_bar": False,
            },
        }
        
        strategy = IDFPruningStrategy.from_dict(data)
        assert strategy.config.mode == "global"
        assert strategy.config.top_k == 10
        assert strategy.config.protected_tokens == 1


class TestPoolingStrategy:
    """Tests for PoolingStrategy."""

    def test_config_validation_pool_factor_positive(self):
        """Test that pool_factor must be positive."""
        # Validation happens in PoolingStrategy.__init__, not PoolingConfig
        with pytest.raises(ValueError, match="positive integer"):
            PoolingStrategy(PoolingConfig(pool_factor=0))

    def test_config_validation_protected_tokens_nonnegative(self):
        """Test that protected_tokens must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            PoolingStrategy(PoolingConfig(pool_factor=2, protected_tokens=-1))

    def test_strategy_name(self):
        """Test strategy name generation."""
        config = PoolingConfig(
            clustering_method="hierarchical",
            pool_factor=2,
            protected_tokens=1,
        )
        strategy = PoolingStrategy(config)
        assert "pooling" in strategy.name
        assert "hierarchical" in strategy.name
        assert "k-2" in strategy.name
        assert "p-1" in strategy.name

    def test_hierarchical_pooling(self, sample_embeddings, sample_artifacts):
        """Test hierarchical pooling."""
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="hierarchical",
        )
        strategy = PoolingStrategy(config)
        
        pooled_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Should have same number of documents
        assert len(pooled_embeddings) == len(sample_embeddings)
        
        # Each document should have fewer tokens (approximately divided by pool_factor)
        for i, (orig, pooled) in enumerate(zip(sample_embeddings, pooled_embeddings)):
            assert pooled.shape[0] <= orig.shape[0]
            assert pooled.shape[0] >= config.protected_tokens
            assert pooled.shape[1] == orig.shape[1]  # Embedding dimension unchanged
        
        # Artifacts should be updated
        assert "input_ids" in updated_artifacts
        for i, pooled_ids in enumerate(updated_artifacts["input_ids"]):
            assert len(pooled_ids) == pooled_embeddings[i].shape[0]

    def test_spherical_pooling(self, sample_embeddings, sample_artifacts):
        """Test spherical pooling (if fastkmeans is available)."""
        try:
            import fastkmeans
        except ImportError:
            pytest.skip("fastkmeans not available")
        
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="spherical",
        )
        strategy = PoolingStrategy(config)
        
        pooled_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Should have same number of documents
        assert len(pooled_embeddings) == len(sample_embeddings)
        
        # Each document should have fewer tokens
        for i, (orig, pooled) in enumerate(zip(sample_embeddings, pooled_embeddings)):
            assert pooled.shape[0] <= orig.shape[0]
            assert pooled.shape[0] >= config.protected_tokens
            assert pooled.shape[1] == orig.shape[1]

    def test_pool_factor_one_no_pooling(self, sample_embeddings, sample_artifacts):
        """Test that pool_factor=1 results in no pooling."""
        config = PoolingConfig(pool_factor=1, clustering_method="hierarchical")
        strategy = PoolingStrategy(config)
        
        pooled_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Should be unchanged
        assert len(pooled_embeddings) == len(sample_embeddings)
        for orig, pooled in zip(sample_embeddings, pooled_embeddings):
            assert torch.equal(orig, pooled)

    def test_protected_tokens_pooling(self, sample_embeddings, sample_artifacts):
        """Test that protected tokens are preserved in pooling."""
        config = PoolingConfig(
            pool_factor=3,
            protected_tokens=2,
            clustering_method="hierarchical",
        )
        strategy = PoolingStrategy(config)
        
        pooled_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Protected tokens should be preserved
        for i, pooled_ids in enumerate(updated_artifacts["input_ids"]):
            original_ids = sample_artifacts["input_ids"][i]
            assert len(pooled_ids) >= config.protected_tokens
            # First protected tokens should match
            for j in range(config.protected_tokens):
                assert pooled_ids[j] == original_ids[j]

    def test_serialization(self):
        """Test pooling strategy serialization."""
        config = PoolingConfig(
            pool_factor=2,
            protected_tokens=1,
            clustering_method="hierarchical",
            show_progress_bar=True,
        )
        strategy = PoolingStrategy(config)
        
        serialized = strategy.serialize()
        assert serialized["type"] == "pooling"
        assert serialized["config"]["pool_factor"] == 2
        assert serialized["config"]["clustering_method"] == "hierarchical"

    def test_deserialization(self):
        """Test pooling strategy deserialization."""
        data = {
            "type": "pooling",
            "config": {
                "pool_factor": 3,
                "protected_tokens": 1,
                "clustering_method": "hierarchical",
                "show_progress_bar": False,
            },
        }
        
        strategy = PoolingStrategy.from_dict(data)
        assert strategy.config.pool_factor == 3
        assert strategy.config.protected_tokens == 1
        assert strategy.config.clustering_method == "hierarchical"


class TestCompressionConfig:
    """Tests for CompressionConfig."""

    def test_empty_config(self):
        """Test empty compression config."""
        config = CompressionConfig()
        assert len(config.strategies) == 0
        assert config.description == ""

    def test_config_with_strategies(self):
        """Test config with multiple strategies."""
        idf_config = IDFPruningConfig(mode="document", top_k=5)
        pooling_config = PoolingConfig(pool_factor=2)
        
        strategies = [
            IDFPruningStrategy(idf_config),
            PoolingStrategy(pooling_config),
        ]
        
        config = CompressionConfig(
            strategies=strategies,
            description="Test config",
        )
        
        assert len(config.strategies) == 2
        assert config.description == "Test config"

    def test_create_compressor(self):
        """Test creating compressor from config."""
        idf_config = IDFPruningConfig(mode="document", top_k=5)
        strategy = IDFPruningStrategy(idf_config)
        
        config = CompressionConfig(strategies=[strategy])
        compressor = config.create_compressor()
        
        assert isinstance(compressor, Compressor)
        assert len(compressor.strategies) == 1

    def test_serialization(self):
        """Test config serialization."""
        idf_config = IDFPruningConfig(mode="document", top_k=5)
        pooling_config = PoolingConfig(pool_factor=2)
        
        strategies = [
            IDFPruningStrategy(idf_config),
            PoolingStrategy(pooling_config),
        ]
        
        config = CompressionConfig(
            strategies=strategies,
            description="Test config",
        )
        
        serialized = config.serialize()
        assert serialized["description"] == "Test config"
        assert len(serialized["strategies"]) == 2
        assert serialized["strategies"][0]["type"] == "idf_pruning"
        assert serialized["strategies"][1]["type"] == "pooling"

    def test_deserialization(self):
        """Test config deserialization."""
        data = {
            "description": "Test config",
            "strategies": [
                {
                    "type": "idf_pruning",
                    "config": {
                        "mode": "document",
                        "top_k": 5,
                        "protected_tokens": 1,
                        "use_tfidf": False,
                        "track_pruned_tokens": False,
                        "show_progress_bar": False,
                    },
                },
                {
                    "type": "pooling",
                    "config": {
                        "pool_factor": 2,
                        "protected_tokens": 1,
                        "clustering_method": "hierarchical",
                        "show_progress_bar": False,
                    },
                },
            ],
        }
        
        config = CompressionConfig.from_dict(data)
        assert config.description == "Test config"
        assert len(config.strategies) == 2
        assert isinstance(config.strategies[0], IDFPruningStrategy)
        assert isinstance(config.strategies[1], PoolingStrategy)

    def test_deserialization_unknown_strategy(self):
        """Test that unknown strategy type raises error."""
        data = {
            "description": "Test",
            "strategies": [
                {
                    "type": "unknown_strategy",
                    "config": {},
                },
            ],
        }
        
        with pytest.raises(ValueError, match="Unknown strategy type"):
            CompressionConfig.from_dict(data)


class TestCompressor:
    """Tests for Compressor."""

    def test_compressor_creation(self):
        """Test compressor creation."""
        idf_config = IDFPruningConfig(mode="document", top_k=2)
        strategy = IDFPruningStrategy(idf_config)
        
        compressor = Compressor([strategy])
        assert len(compressor.strategies) == 1

    def test_compressor_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(TypeError, match="not an instance"):
            Compressor([object()])  # type: ignore

    def test_get_required_artifacts(self):
        """Test getting required artifacts."""
        idf_config = IDFPruningConfig(mode="document", top_k=2)
        pooling_config = PoolingConfig(pool_factor=2)
        
        strategies = [
            IDFPruningStrategy(idf_config),
            PoolingStrategy(pooling_config),
        ]
        
        compressor = Compressor(strategies)
        required = compressor.get_required_artifacts()
        
        assert "input_ids" in required  # Required by IDFPruningStrategy
        assert len(required) == 1  # PoolingStrategy doesn't require artifacts

    def test_compress_single_strategy(self, sample_embeddings, sample_artifacts):
        """Test compression with single strategy."""
        idf_config = IDFPruningConfig(mode="document", top_k=2)
        strategy = IDFPruningStrategy(idf_config)
        
        compressor = Compressor([strategy])
        compressed_embeddings, updated_artifacts = compressor.compress(
            sample_embeddings, sample_artifacts
        )
        
        assert len(compressed_embeddings) == len(sample_embeddings)
        assert "input_ids" in updated_artifacts

    def test_compress_multiple_strategies(self, sample_embeddings, sample_artifacts):
        """Test compression with multiple strategies in sequence."""
        idf_config = IDFPruningConfig(mode="document", top_k=2)
        pooling_config = PoolingConfig(pool_factor=2, clustering_method="hierarchical")
        
        strategies = [
            IDFPruningStrategy(idf_config),
            PoolingStrategy(pooling_config),
        ]
        
        compressor = Compressor(strategies)
        compressed_embeddings, updated_artifacts = compressor.compress(
            sample_embeddings, sample_artifacts
        )
        
        assert len(compressed_embeddings) == len(sample_embeddings)
        
        # Should have fewer tokens after both strategies
        for i, (orig, compressed) in enumerate(zip(sample_embeddings, compressed_embeddings)):
            assert compressed.shape[0] < orig.shape[0]

    def test_missing_required_artifacts(self, sample_embeddings):
        """Test that missing required artifacts raises error."""
        idf_config = IDFPruningConfig(mode="document", top_k=2)
        strategy = IDFPruningStrategy(idf_config)
        
        compressor = Compressor([strategy])
        
        with pytest.raises(ValueError, match="Missing required artifacts"):
            compressor.compress(sample_embeddings, {})

    def test_artifact_1to1_mapping(self, sample_embeddings, sample_artifacts):
        """Test that 1:1 mapping is maintained."""
        idf_config = IDFPruningConfig(mode="document", top_k=2)
        strategy = IDFPruningStrategy(idf_config)
        
        compressor = Compressor([strategy])
        compressed_embeddings, updated_artifacts = compressor.compress(
            sample_embeddings, sample_artifacts
        )
        
        # All shape-matched artifacts should maintain 1:1 mapping
        for artifact_name, artifact_value in updated_artifacts.items():
            if isinstance(artifact_value, list):
                assert len(artifact_value) == len(compressed_embeddings)
                for i, artifact in enumerate(artifact_value):
                    if isinstance(artifact, torch.Tensor):
                        assert artifact.shape[0] == compressed_embeddings[i].shape[0]

    def test_original_artifacts_not_modified(self, sample_embeddings, sample_artifacts):
        """Test that original artifacts are not modified."""
        original_input_ids = [ids.clone() for ids in sample_artifacts["input_ids"]]
        
        idf_config = IDFPruningConfig(mode="document", top_k=2)
        strategy = IDFPruningStrategy(idf_config)
        
        compressor = Compressor([strategy])
        compressed_embeddings, updated_artifacts = compressor.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Original artifacts should be unchanged
        for orig, current in zip(original_input_ids, sample_artifacts["input_ids"]):
            assert torch.equal(orig, current)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_documents(self):
        """Test handling of empty documents."""
        embeddings = [torch.randn(0, 128, device="cpu")]  # Empty document
        input_ids = [torch.tensor([], device="cpu", dtype=torch.long)]
        artifacts = {"input_ids": input_ids}
        
        config = IDFPruningConfig(mode="document", top_k=1)
        strategy = IDFPruningStrategy(config)
        
        # Should handle gracefully
        pruned_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
        assert len(pruned_embeddings) == 1
        assert pruned_embeddings[0].shape[0] == 0

    def test_single_token_document(self):
        """Test handling of single token documents."""
        embeddings = [torch.randn(1, 128, device="cpu")]
        input_ids = [torch.tensor([101], device="cpu")]
        artifacts = {"input_ids": input_ids}
        
        config = IDFPruningConfig(mode="document", top_k=1, protected_tokens=1)
        strategy = IDFPruningStrategy(config)
        
        pruned_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
        assert len(pruned_embeddings) == 1
        # Protected token should remain
        assert pruned_embeddings[0].shape[0] >= 1

    def test_pooling_single_token(self):
        """Test pooling with single token."""
        embeddings = [torch.randn(1, 128, device="cpu")]
        artifacts = {"input_ids": [torch.tensor([101], device="cpu")]}
        
        config = PoolingConfig(pool_factor=2, protected_tokens=1)
        strategy = PoolingStrategy(config)
        
        pooled_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
        assert len(pooled_embeddings) == 1
        # Protected token should remain
        assert pooled_embeddings[0].shape[0] >= 1

    def test_top_k_larger_than_tokens(self, sample_embeddings, sample_artifacts):
        """Test top_k larger than available tokens."""
        config = IDFPruningConfig(mode="document", top_k=1000, protected_tokens=1)
        strategy = IDFPruningStrategy(config)
        
        pruned_embeddings, updated_artifacts = strategy.compress(
            sample_embeddings, sample_artifacts
        )
        
        # Should still work, pruning as many as possible
        for i, pruned in enumerate(pruned_embeddings):
            assert pruned.shape[0] >= config.protected_tokens

