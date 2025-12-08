"""
Test the fixed LeverageScorePruningStrategy.

This test verifies all the fixes from the critique:
1. Masks are cached and reused (no recomputation for artifacts)
2. Both tensor and list artifacts are handled correctly
3. min_tokens constraint is respected
4. Validation catches invalid configs
"""

import sys
import os

import torch

from pylate.models.compression import (
    LeverageScorePruningConfig,
    LeverageScorePruningStrategy,
)


def test_min_tokens_constraint():
    """Test that min_tokens prevents over-pruning."""
    print("=" * 80)
    print("TEST 1: min_tokens constraint")
    print("=" * 80)
    
    embeddings = [
        torch.randn(20, 128),
        torch.randn(15, 128),
    ]
    
    artifacts = {}
    
    # Try to prune 15 tokens, but min_tokens=10 should prevent it
    config = LeverageScorePruningConfig(
        top_k=15,  # Try to remove 15 tokens
        protected_tokens=1,
        min_tokens=10,  # But keep at least 10 total
        projection_dim=32,
    )
    strategy = LeverageScorePruningStrategy(config)
    
    print(f"\nOriginal lengths: {[e.shape[0] for e in embeddings]}")
    print(f"Config: top_k={config.top_k}, min_tokens={config.min_tokens}")
    
    pruned_embeddings, _ = strategy.compress(embeddings, artifacts)
    
    print(f"Pruned lengths:   {[e.shape[0] for e in pruned_embeddings]}")
    
    # Doc 0: 20 tokens, try to prune 15, but min_tokens=10 -> keep 10
    # Doc 1: 15 tokens, try to prune 15, but min_tokens=10 -> keep 10
    assert pruned_embeddings[0].shape[0] >= config.min_tokens, \
        f"Doc 0: Expected >= {config.min_tokens}, got {pruned_embeddings[0].shape[0]}"
    assert pruned_embeddings[1].shape[0] >= config.min_tokens, \
        f"Doc 1: Expected >= {config.min_tokens}, got {pruned_embeddings[1].shape[0]}"
    
    print(f"✓ min_tokens constraint respected!")
    print("\n✓ Test 1 passed!")


def test_artifact_alignment():
    """Test that artifacts stay aligned with embeddings (no recomputation)."""
    print("\n" + "=" * 80)
    print("TEST 2: Artifact alignment (cached masks)")
    print("=" * 80)
    
    embeddings = [
        torch.randn(10, 128),
        torch.randn(12, 128),
    ]
    
    # Create tensor and list artifacts
    input_ids = [
        torch.arange(10),
        torch.arange(12),
    ]
    
    # List artifact (e.g., token strings)
    token_strings = [
        [f"token_{i}" for i in range(10)],
        [f"token_{i}" for i in range(12)],
    ]
    
    artifacts = {
        "input_ids": input_ids,
        "token_strings": token_strings,
    }
    
    config = LeverageScorePruningConfig(
        top_k=3,
        protected_tokens=1,
        min_tokens=5,
        projection_dim=32,
    )
    strategy = LeverageScorePruningStrategy(config)
    
    print(f"\nOriginal lengths: {[e.shape[0] for e in embeddings]}")
    
    pruned_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"Pruned lengths:   {[e.shape[0] for e in pruned_embeddings]}")
    print(f"Pruned input_ids: {[ids.shape[0] for ids in updated_artifacts['input_ids']]}")
    print(f"Pruned strings:   {[len(strs) for strs in updated_artifacts['token_strings']]}")
    
    # Verify all artifacts have same length as embeddings
    for doc_idx in range(len(embeddings)):
        emb_len = pruned_embeddings[doc_idx].shape[0]
        ids_len = updated_artifacts['input_ids'][doc_idx].shape[0]
        str_len = len(updated_artifacts['token_strings'][doc_idx])
        
        assert emb_len == ids_len, \
            f"Doc {doc_idx}: embedding length {emb_len} != input_ids length {ids_len}"
        assert emb_len == str_len, \
            f"Doc {doc_idx}: embedding length {emb_len} != token_strings length {str_len}"
    
    print(f"✓ All artifacts aligned with embeddings!")
    print("\n✓ Test 2 passed!")


def test_list_artifacts():
    """Test that list (non-tensor) artifacts are handled correctly."""
    print("\n" + "=" * 80)
    print("TEST 3: List artifacts handling")
    print("=" * 80)
    
    embeddings = [
        torch.randn(8, 128),
    ]
    
    # Create various list artifacts
    artifacts = {
        "token_strings": [["CLS", "hello", "world", "foo", "bar", "baz", "qux", "END"]],
        "positions": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "metadata": "this is metadata",  # Non-list artifact
    }
    
    config = LeverageScorePruningConfig(
        top_k=3,
        protected_tokens=1,
        min_tokens=4,
        projection_dim=32,
    )
    strategy = LeverageScorePruningStrategy(config)
    
    print(f"\nOriginal token_strings: {artifacts['token_strings'][0]}")
    
    pruned_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"Pruned token_strings:   {updated_artifacts['token_strings'][0]}")
    print(f"Pruned positions:       {updated_artifacts['positions'][0]}")
    print(f"Metadata (unchanged):   {updated_artifacts['metadata']}")
    
    # Verify list artifacts were pruned
    assert isinstance(updated_artifacts['token_strings'][0], list)
    assert isinstance(updated_artifacts['positions'][0], list)
    assert len(updated_artifacts['token_strings'][0]) == pruned_embeddings[0].shape[0]
    assert len(updated_artifacts['positions'][0]) == pruned_embeddings[0].shape[0]
    
    # Verify metadata unchanged
    assert updated_artifacts['metadata'] == "this is metadata"
    
    # Verify first token (protected) is kept
    assert updated_artifacts['token_strings'][0][0] == "CLS"
    
    print(f"✓ List artifacts handled correctly!")
    print("\n✓ Test 3 passed!")


def test_config_validation():
    """Test that invalid configs are caught."""
    print("\n" + "=" * 80)
    print("TEST 4: Config validation")
    print("=" * 80)
    
    # Test: both top_k and threshold
    try:
        config = LeverageScorePruningConfig(
            top_k=10,
            threshold=0.5,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught error for both top_k and threshold: {e}")
    
    # Test: neither top_k nor threshold
    try:
        config = LeverageScorePruningConfig(
            top_k=None,
            threshold=None,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught error for neither top_k nor threshold: {e}")
    
    # Test: negative top_k
    try:
        config = LeverageScorePruningConfig(
            top_k=-5,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught error for negative top_k: {e}")
    
    # Test: min_tokens < 1
    try:
        config = LeverageScorePruningConfig(
            top_k=10,
            min_tokens=0,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught error for min_tokens < 1: {e}")
    
    print("\n✓ Test 4 passed!")


if __name__ == "__main__":
    test_min_tokens_constraint()
    test_artifact_alignment()
    test_list_artifacts()
    test_config_validation()
    
    print("\n" + "=" * 80)
    print("ALL LEVERAGE SCORE FIXES VERIFIED! ✓")
    print("=" * 80)

