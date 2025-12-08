"""
Tests for the HybridImportanceClusteringPoolingStrategy.

Matches the style of other compression tests: print progress + assertions.
"""

import torch

from pylate.models.compression import (
    CompressionConfig,
    HybridPoolingConfig,
    HybridImportanceClusteringPoolingStrategy,
)


def test_hybrid_pooling_basic():
    """Hybrid pooling should reduce tokens while keeping artifacts aligned."""
    print("=" * 80)
    print("TEST 1: Hybrid pooling basic")
    print("=" * 80)

    embeddings = [
        torch.randn(16, 32),
        torch.randn(10, 32),
    ]
    artifacts = {
        "input_ids": [
            torch.arange(16),
            torch.arange(10),
        ],
    }

    config = HybridPoolingConfig(
        pool_factor=3,
        keep_ratio=0.5,
        protected_tokens=1,
        min_tokens=6,
        clustering_method="hierarchical",
        use_norm=True,
        use_idf=False,
        use_token_weights=False,
    )
    strategy = HybridImportanceClusteringPoolingStrategy(config)

    print(f"\nOriginal lengths: {[e.shape[0] for e in embeddings]}")
    print(f"Config: pf={config.pool_factor}, kr={config.keep_ratio}, min={config.min_tokens}")

    pooled_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)

    print(f"Pooled lengths:   {[e.shape[0] for e in pooled_embeddings]}")
    print(f"Pooled input_ids: {[ids.shape[0] for ids in updated_artifacts['input_ids']]}")

    for idx in range(len(embeddings)):
        assert pooled_embeddings[idx].shape[0] <= embeddings[idx].shape[0]
        assert pooled_embeddings[idx].shape[0] >= config.min_tokens
        assert (
            pooled_embeddings[idx].shape[0]
            == updated_artifacts["input_ids"][idx].shape[0]
        )
        # protected tokens stay untouched
        assert torch.equal(
            updated_artifacts["input_ids"][idx][: config.protected_tokens],
            artifacts["input_ids"][idx][: config.protected_tokens],
        )

    print("✓ Hybrid pooling basic passed!")


def test_hybrid_serialization_round_trip():
    """Serialization should preserve config and be usable in CompressionConfig."""
    print("\n" + "=" * 80)
    print("TEST 2: Hybrid serialization")
    print("=" * 80)

    config = HybridPoolingConfig(
        pool_factor=2,
        keep_ratio=0.6,
        protected_tokens=2,
        min_tokens=5,
        clustering_method="hierarchical",
        show_progress_bar=False,
        use_norm=True,
        use_idf=True,
        use_token_weights=True,
        norm_weight=1.3,
        idf_weight=0.7,
        token_weights_weight=0.9,
    )
    strategy = HybridImportanceClusteringPoolingStrategy(config)

    serialized = strategy.serialize()
    print(f"\nSerialized: {serialized}")
    restored = HybridImportanceClusteringPoolingStrategy.from_dict(serialized)

    assert restored.config.pool_factor == config.pool_factor
    assert restored.config.keep_ratio == config.keep_ratio
    assert restored.config.protected_tokens == config.protected_tokens
    assert restored.config.min_tokens == config.min_tokens
    assert restored.config.clustering_method == config.clustering_method
    assert restored.config.use_idf == config.use_idf
    assert restored.config.use_token_weights == config.use_token_weights
    assert restored.config.norm_weight == config.norm_weight

    comp_config = CompressionConfig(
        strategies=[strategy],
        description="Hybrid test",
    )
    comp_serialized = comp_config.serialize()
    comp_round_tripped = CompressionConfig.from_dict(comp_serialized)

    assert len(comp_round_tripped.strategies) == 1
    assert isinstance(
        comp_round_tripped.strategies[0], HybridImportanceClusteringPoolingStrategy
    )

    print("✓ Hybrid serialization passed!")


if __name__ == "__main__": 
    test_hybrid_pooling_basic()
    test_hybrid_serialization_round_trip()
    
    print("\n" + "=" * 80)
    print("ALL HYBRID IMPORTANCE POOLING TESTS PASSED! ✓")
    print("=" * 80)

