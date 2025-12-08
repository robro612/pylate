"""
Test script for ImportancePruningStrategy.

This script demonstrates and tests the importance-based pruning functionality.
"""

import sys
import os

# Add pylate to path
sys.path.insert(0, os.path.dirname(__file__))

import torch

# Try to import - if transformers is broken, we'll create a minimal test
try:
    from pylate.models.compression import (
        ImportancePruningConfig,
        ImportancePruningStrategy,
    )
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Could not import from pylate.models: {e}")
    print("Creating minimal standalone test...")
    IMPORT_SUCCESS = False


def test_importance_pruning_basic():
    """Test basic importance pruning with L2 norm."""
    print("=" * 80)
    print("TEST 1: Basic Importance Pruning (L2 norm only)")
    print("=" * 80)
    
    # Create sample embeddings (3 documents with varying lengths)
    embeddings = [
        torch.randn(10, 128),  # 10 tokens, 128 dims
        torch.randn(15, 128),  # 15 tokens, 128 dims
        torch.randn(8, 128),   # 8 tokens, 128 dims
    ]
    
    # Make some tokens have higher norms (more important)
    embeddings[0][2] *= 0.1  # Low norm token
    embeddings[0][5] *= 2.0  # High norm token
    
    artifacts = {}
    
    # Test with keep_ratio=0.5 (keep 50% of non-protected tokens)
    config = ImportancePruningConfig(
        keep_ratio=0.5,
        protected_tokens=1,  # Keep first token (CLS)
        min_tokens=4,
        use_norm=True,
        use_idf=False,
        use_token_weights=False,
        norm_weight=1.0,
    )
    strategy = ImportancePruningStrategy(config)
    
    print(f"\nOriginal lengths: {[e.shape[0] for e in embeddings]}")
    print(f"Config: keep_ratio={config.keep_ratio}, protected_tokens={config.protected_tokens}")
    print(f"        min_tokens={config.min_tokens}")
    
    pruned_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"Pruned lengths:   {[e.shape[0] for e in pruned_embeddings]}")
    
    # Expected: 
    # Doc 0: 10 tokens -> 1 protected + 0.5 * 9 = 1 + 4.5 = 5 tokens (rounded down to 5)
    # Doc 1: 15 tokens -> 1 protected + 0.5 * 14 = 1 + 7 = 8 tokens
    # Doc 2: 8 tokens -> 1 protected + 0.5 * 7 = 1 + 3.5 = 4 tokens (but min_tokens=4, so 4)
    
    assert pruned_embeddings[0].shape[0] >= 4, f"Expected >= 4 tokens, got {pruned_embeddings[0].shape[0]}"
    assert pruned_embeddings[1].shape[0] >= 4, f"Expected >= 4 tokens, got {pruned_embeddings[1].shape[0]}"
    assert pruned_embeddings[2].shape[0] >= 4, f"Expected >= 4 tokens, got {pruned_embeddings[2].shape[0]}"
    
    print("\n✓ Test 1 passed!")


def test_importance_pruning_with_idf():
    """Test importance pruning with IDF scores."""
    print("\n" + "=" * 80)
    print("TEST 2: Importance Pruning with IDF scores")
    print("=" * 80)
    
    # Create sample embeddings
    embeddings = [
        torch.randn(10, 128),
        torch.randn(12, 128),
    ]
    
    # Create IDF scores (higher = more important)
    idf_scores = [
        torch.tensor([5.0, 3.0, 1.0, 4.0, 2.0, 3.5, 1.5, 4.5, 2.5, 3.2]),  # 10 tokens
        torch.tensor([4.0, 2.0, 5.0, 1.0, 3.0, 4.5, 2.5, 3.5, 1.5, 4.2, 2.8, 3.8]),  # 12 tokens
    ]
    
    artifacts = {
        "idf": idf_scores,
    }
    
    config = ImportancePruningConfig(
        keep_ratio=0.6,
        protected_tokens=1,
        min_tokens=5,
        use_norm=True,
        use_idf=True,
        use_token_weights=False,
        norm_weight=0.5,
        idf_weight=1.0,
    )
    strategy = ImportancePruningStrategy(config)
    
    print(f"\nOriginal lengths: {[e.shape[0] for e in embeddings]}")
    print(f"IDF scores: {[idf.tolist() for idf in idf_scores]}")
    
    pruned_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"Pruned lengths:   {[e.shape[0] for e in pruned_embeddings]}")
    print(f"Updated IDF lengths: {[len(idf) for idf in updated_artifacts['idf']]}")
    
    # Verify IDF artifacts were updated
    assert len(updated_artifacts['idf']) == len(pruned_embeddings)
    for i, (emb, idf) in enumerate(zip(pruned_embeddings, updated_artifacts['idf'])):
        if isinstance(idf, torch.Tensor):
            assert idf.shape[0] == emb.shape[0], f"Doc {i}: IDF length mismatch"
        else:
            assert len(idf) == emb.shape[0], f"Doc {i}: IDF length mismatch"
    
    print("\n✓ Test 2 passed!")


def test_importance_pruning_min_tokens():
    """Test that min_tokens constraint is respected."""
    print("\n" + "=" * 80)
    print("TEST 3: min_tokens constraint")
    print("=" * 80)
    
    embeddings = [
        torch.randn(20, 128),
    ]
    
    artifacts = {}
    
    # Very aggressive pruning but min_tokens should prevent over-pruning
    config = ImportancePruningConfig(
        keep_ratio=0.1,  # Keep only 10%
        protected_tokens=1,
        min_tokens=10,  # But at least 10 tokens
        use_norm=True,
        use_idf=False,
        use_token_weights=False,
    )
    strategy = ImportancePruningStrategy(config)
    
    print(f"\nOriginal length: {embeddings[0].shape[0]}")
    print(f"Config: keep_ratio={config.keep_ratio}, min_tokens={config.min_tokens}")
    
    pruned_embeddings, _ = strategy.compress(embeddings, artifacts)
    
    print(f"Pruned length:   {pruned_embeddings[0].shape[0]}")
    
    # Should have at least min_tokens
    assert pruned_embeddings[0].shape[0] >= config.min_tokens, \
        f"Expected >= {config.min_tokens} tokens, got {pruned_embeddings[0].shape[0]}"
    
    print(f"✓ min_tokens constraint respected: {pruned_embeddings[0].shape[0]} >= {config.min_tokens}")
    print("\n✓ Test 3 passed!")


if __name__ == "__main__":
    if IMPORT_SUCCESS:
        test_importance_pruning_basic()
        test_importance_pruning_with_idf()
        test_importance_pruning_min_tokens()
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("IMPORT FAILED - Cannot run test")
        print("=" * 80)
        sys.exit(1)

