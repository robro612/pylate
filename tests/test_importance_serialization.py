"""
Test serialization/deserialization of ImportancePruningStrategy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pylate.models.compression import (
    ImportancePruningConfig,
    ImportancePruningStrategy,
    CompressionConfig,
)


def test_strategy_serialization():
    """Test that ImportancePruningStrategy can be serialized and deserialized."""
    print("=" * 80)
    print("TEST: ImportancePruningStrategy Serialization/Deserialization")
    print("=" * 80)
    
    # Create a strategy
    config = ImportancePruningConfig(
        keep_ratio=0.6,
        protected_tokens=2,
        min_tokens=5,
        use_norm=True,
        use_idf=True,
        use_token_weights=False,
        norm_weight=1.5,
        idf_weight=2.0,
        token_weights_weight=0.5,
    )
    strategy = ImportancePruningStrategy(config)
    
    print("\nOriginal strategy:")
    print(f"  Name: {strategy.name}")
    print(f"  Type: {strategy.strategy_type}")
    print(f"  Config: {config}")
    
    # Serialize
    serialized = strategy.serialize()
    print(f"\nSerialized: {serialized}")
    
    # Deserialize
    deserialized_strategy = ImportancePruningStrategy.from_dict(serialized)
    
    print(f"\nDeserialized strategy:")
    print(f"  Name: {deserialized_strategy.name}")
    print(f"  Type: {deserialized_strategy.strategy_type}")
    print(f"  Config: {deserialized_strategy.config}")
    
    # Verify
    assert deserialized_strategy.strategy_type == strategy.strategy_type
    assert deserialized_strategy.config.keep_ratio == config.keep_ratio
    assert deserialized_strategy.config.protected_tokens == config.protected_tokens
    assert deserialized_strategy.config.min_tokens == config.min_tokens
    assert deserialized_strategy.config.use_norm == config.use_norm
    assert deserialized_strategy.config.use_idf == config.use_idf
    assert deserialized_strategy.config.use_token_weights == config.use_token_weights
    assert deserialized_strategy.config.norm_weight == config.norm_weight
    assert deserialized_strategy.config.idf_weight == config.idf_weight
    assert deserialized_strategy.config.token_weights_weight == config.token_weights_weight
    
    print("\n✓ Serialization/deserialization successful!")


def test_compression_config_with_importance():
    """Test that CompressionConfig can serialize/deserialize with ImportancePruningStrategy."""
    print("\n" + "=" * 80)
    print("TEST: CompressionConfig with ImportancePruningStrategy")
    print("=" * 80)
    
    # Create a compression config with importance pruning
    importance_config = ImportancePruningConfig(
        keep_ratio=0.7,
        protected_tokens=1,
        min_tokens=8,
    )
    strategy = ImportancePruningStrategy(importance_config)
    
    compression_config = CompressionConfig(
        strategies=[strategy],
        description="Test importance pruning config",
    )
    
    print(f"\nOriginal CompressionConfig:")
    print(f"  Description: {compression_config.description}")
    print(f"  Strategies: {[s.name for s in compression_config.strategies]}")
    
    # Serialize
    serialized = compression_config.serialize()
    print(f"\nSerialized: {serialized}")
    
    # Deserialize
    deserialized_config = CompressionConfig.from_dict(serialized)
    
    print(f"\nDeserialized CompressionConfig:")
    print(f"  Description: {deserialized_config.description}")
    print(f"  Strategies: {[s.name for s in deserialized_config.strategies]}")
    
    # Verify
    assert len(deserialized_config.strategies) == 1
    assert isinstance(deserialized_config.strategies[0], ImportancePruningStrategy)
    assert deserialized_config.strategies[0].config.keep_ratio == 0.7
    assert deserialized_config.strategies[0].config.protected_tokens == 1
    assert deserialized_config.strategies[0].config.min_tokens == 8
    assert deserialized_config.description == "Test importance pruning config"
    
    print("\n✓ CompressionConfig serialization/deserialization successful!")


if __name__ == "__main__":
    test_strategy_serialization()
    test_compression_config_with_importance()
    print("\n" + "=" * 80)
    print("ALL SERIALIZATION TESTS PASSED! ✓")
    print("=" * 80)

