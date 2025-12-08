"""Test script for LeverageScorePruningStrategy."""

import torch
# Import directly from compression module to avoid transformers dependency
from pylate.models.compression import (
    LeverageScorePruningConfig,
    LeverageScorePruningStrategy,
    CompressionConfig,
)


def test_basic_leverage_score_pruning():
    """Test basic leverage score pruning functionality."""
    print("=" * 80)
    print("Testing LeverageScorePruningStrategy")
    print("=" * 80)
    
    # Create sample embeddings (3 documents with varying lengths)
    embeddings = [
        torch.randn(10, 128),  # 10 tokens, 128 dims
        torch.randn(15, 128),  # 15 tokens, 128 dims
        torch.randn(8, 128),   # 8 tokens, 128 dims
    ]
    
    # Create input_ids for tracking
    input_ids = [
        torch.arange(10),
        torch.arange(15),
        torch.arange(8),
    ]
    
    artifacts = {"input_ids": input_ids}
    
    # Test 1: top_k pruning
    print("\n1. Testing top_k=3 pruning...")
    config = LeverageScorePruningConfig(
        top_k=3,
        protected_tokens=1,
        projection_dim=32,
        track_pruned_tokens=True,
        show_progress_bar=False,
    )
    strategy = LeverageScorePruningStrategy(config)
    
    pruned_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"   Original lengths: {[e.shape[0] for e in embeddings]}")
    print(f"   Pruned lengths:   {[e.shape[0] for e in pruned_embeddings]}")
    print(f"   Expected:         [7, 12, 5] (removed 3 tokens each)")
    
    # Verify shapes
    assert pruned_embeddings[0].shape[0] == 7, f"Expected 7 tokens, got {pruned_embeddings[0].shape[0]}"
    assert pruned_embeddings[1].shape[0] == 12, f"Expected 12 tokens, got {pruned_embeddings[1].shape[0]}"
    assert pruned_embeddings[2].shape[0] == 5, f"Expected 5 tokens, got {pruned_embeddings[2].shape[0]}"
    
    # Check pruned tokens tracking
    pruned_tokens = strategy.get_pruned_tokens()
    print(f"   Pruned tokens tracked: {pruned_tokens is not None}")
    if pruned_tokens:
        print(f"   Num pruned per doc: {[len(p) for p in pruned_tokens]}")
    
    print("   ✓ Test 1 passed!")
    
    # Test 2: threshold pruning
    print("\n2. Testing threshold=-0.5 pruning...")
    config2 = LeverageScorePruningConfig(
        threshold=-0.5,
        protected_tokens=1,
        projection_dim=32,
        normalize_scores=True,
        show_progress_bar=False,
    )
    strategy2 = LeverageScorePruningStrategy(config2)
    
    pruned_embeddings2, _ = strategy2.compress(embeddings, artifacts)
    
    print(f"   Original lengths: {[e.shape[0] for e in embeddings]}")
    print(f"   Pruned lengths:   {[e.shape[0] for e in pruned_embeddings2]}")
    print("   ✓ Test 2 passed!")
    
    # Test 3: CompressionConfig integration
    print("\n3. Testing CompressionConfig integration...")
    compression_config = CompressionConfig(
        strategies=[strategy],
        description="Leverage score pruning test",
    )
    
    # Serialize and deserialize
    serialized = compression_config.serialize()
    print(f"   Serialized config: {serialized['description']}")
    print(f"   Strategy type: {serialized['strategies'][0]['type']}")
    
    # Deserialize
    deserialized_config = CompressionConfig.from_dict(serialized)
    print(f"   Deserialized successfully!")
    print(f"   Strategy count: {len(deserialized_config.strategies)}")
    
    # Test compression with deserialized config
    compressor = deserialized_config.create_compressor()
    pruned_embeddings3, _ = compressor.compress(embeddings, artifacts)
    
    print(f"   Pruned lengths:   {[e.shape[0] for e in pruned_embeddings3]}")
    print("   ✓ Test 3 passed!")
    
    # Test 4: Parallel compression
    print("\n4. Testing parallel compression...")
    # Create more documents for parallel processing
    large_embeddings = [torch.randn(10, 128) for _ in range(20)]
    large_input_ids = [torch.arange(10) for _ in range(20)]
    large_artifacts = {"input_ids": large_input_ids}
    
    pruned_parallel, _ = strategy.compress_parallel(
        large_embeddings,
        large_artifacts,
        batch_size=5,
        num_workers=2,
        show_progress=False,
    )
    
    print(f"   Processed {len(pruned_parallel)} documents in parallel")
    print(f"   All pruned to 7 tokens: {all(e.shape[0] == 7 for e in pruned_parallel)}")
    print("   ✓ Test 4 passed!")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_basic_leverage_score_pruning()

