"""
Test serialization and deserialization for ImportancePoolingStrategy.
"""

import sys
import os

import torch

from pylate.models.compression import (
    ImportancePoolingConfig,
    ImportancePoolingStrategy,
    CompressionConfig,
)


def test_serialization():
    """Test serialization and deserialization."""
    print("=" * 80)
    print("TEST: Serialization")
    print("=" * 80)
    
    config = ImportancePoolingConfig(
        keep_ratio=0.6,
        protected_tokens=2,
        min_tokens=10,
        use_norm=True,
        use_idf=True,
        use_token_weights=False,
        norm_weight=2.0,
        idf_weight=1.5,
    )
    strategy = ImportancePoolingStrategy(config)
    
    # Serialize
    serialized = strategy.serialize()
    print(f"\nSerialized: {serialized}")
    
    # Deserialize
    deserialized_strategy = ImportancePoolingStrategy.from_dict(serialized)
    
    # Verify config matches
    assert deserialized_strategy.config.keep_ratio == config.keep_ratio
    assert deserialized_strategy.config.protected_tokens == config.protected_tokens
    assert deserialized_strategy.config.min_tokens == config.min_tokens
    assert deserialized_strategy.config.use_norm == config.use_norm
    assert deserialized_strategy.config.use_idf == config.use_idf
    assert deserialized_strategy.config.norm_weight == config.norm_weight
    assert deserialized_strategy.config.idf_weight == config.idf_weight
    
    print("✓ Serialization works!")
    
    # Test CompressionConfig serialization
    comp_config = CompressionConfig(
        strategies=[strategy],
        description="Test importance pooling",
    )
    comp_serialized = comp_config.serialize()
    comp_deserialized = CompressionConfig.from_dict(comp_serialized)
    
    assert len(comp_deserialized.strategies) == 1
    assert isinstance(comp_deserialized.strategies[0], ImportancePoolingStrategy)
    
    print("✓ CompressionConfig serialization works!")
    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_serialization()
    
    print("\n" + "=" * 80)
    print("SERIALIZATION TEST PASSED! ✓")
    print("=" * 80)

