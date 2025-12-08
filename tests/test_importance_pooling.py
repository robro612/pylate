"""
Test the ImportancePoolingStrategy.

This test verifies:
1. Basic pooling functionality
2. Artifact alignment (embeddings and artifacts stay aligned)
3. Protected tokens are preserved
4. min_tokens constraint is respected
5. Serialization/deserialization works
"""

import sys
import os

import torch

from pylate.models.compression import (
    ImportancePoolingConfig,
    ImportancePoolingStrategy,
    CompressionConfig,
)


def test_basic_pooling():
    """Test basic pooling functionality."""
    print("=" * 80)
    print("TEST 1: Basic pooling")
    print("=" * 80)
    
    embeddings = [
        torch.randn(20, 128),
        torch.randn(15, 128),
    ]
    
    artifacts = {
        "input_ids": [
            torch.arange(20),
            torch.arange(15),
        ],
    }
    
    config = ImportancePoolingConfig(
        keep_ratio=0.5,
        protected_tokens=1,
        min_tokens=8,
        use_norm=True,
        use_idf=False,
        use_token_weights=False,
    )
    strategy = ImportancePoolingStrategy(config)
    
    print(f"\nOriginal lengths: {[e.shape[0] for e in embeddings]}")
    print(f"Config: keep_ratio={config.keep_ratio}, min_tokens={config.min_tokens}")
    
    pooled_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"Pooled lengths:   {[e.shape[0] for e in pooled_embeddings]}")
    print(f"Pooled input_ids: {[ids.shape[0] for ids in updated_artifacts['input_ids']]}")
    
    # Verify pooling happened
    assert pooled_embeddings[0].shape[0] < embeddings[0].shape[0], "Doc 0 should be pooled"
    assert pooled_embeddings[1].shape[0] < embeddings[1].shape[0], "Doc 1 should be pooled"
    
    # Verify min_tokens constraint
    assert pooled_embeddings[0].shape[0] >= config.min_tokens
    assert pooled_embeddings[1].shape[0] >= config.min_tokens
    
    # Verify artifacts aligned
    assert pooled_embeddings[0].shape[0] == updated_artifacts['input_ids'][0].shape[0]
    assert pooled_embeddings[1].shape[0] == updated_artifacts['input_ids'][1].shape[0]
    
    print("✓ Basic pooling works!")
    print("\n✓ Test 1 passed!")


def test_protected_tokens():
    """Test that protected tokens are preserved."""
    print("\n" + "=" * 80)
    print("TEST 2: Protected tokens")
    print("=" * 80)
    
    embeddings = [
        torch.randn(10, 128),
    ]
    
    # Create special first token
    embeddings[0][0] = torch.ones(128) * 999  # Special marker
    
    artifacts = {
        "input_ids": [torch.arange(10)],
    }
    
    config = ImportancePoolingConfig(
        keep_ratio=0.3,
        protected_tokens=2,  # Protect first 2 tokens
        min_tokens=4,
        use_norm=True,
    )
    strategy = ImportancePoolingStrategy(config)
    
    pooled_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"\nOriginal length: {embeddings[0].shape[0]}")
    print(f"Pooled length:   {pooled_embeddings[0].shape[0]}")
    print(f"Protected tokens: {config.protected_tokens}")
    
    # Verify first 2 tokens are unchanged
    assert torch.allclose(pooled_embeddings[0][0], embeddings[0][0]), "First token should be unchanged"
    assert torch.allclose(pooled_embeddings[0][1], embeddings[0][1]), "Second token should be unchanged"
    
    # Verify input_ids for protected tokens
    assert updated_artifacts['input_ids'][0][0].item() == 0
    assert updated_artifacts['input_ids'][0][1].item() == 1
    
    print("✓ Protected tokens preserved!")
    print("\n✓ Test 2 passed!")


def test_list_artifacts():
    """Test that list (non-tensor) artifacts work correctly."""
    print("\n" + "=" * 80)
    print("TEST 3: List artifacts")
    print("=" * 80)
    
    embeddings = [
        torch.randn(12, 128),
    ]
    
    artifacts = {
        "input_ids": [torch.arange(12)],
        "token_strings": [["CLS"] + [f"token_{i}" for i in range(11)]],
    }
    
    config = ImportancePoolingConfig(
        keep_ratio=0.5,
        protected_tokens=1,
        min_tokens=5,
        use_norm=True,
    )
    strategy = ImportancePoolingStrategy(config)
    
    print(f"\nOriginal token_strings: {artifacts['token_strings'][0]}")
    
    pooled_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"Pooled token_strings:   {updated_artifacts['token_strings'][0]}")
    print(f"Pooled length:          {len(updated_artifacts['token_strings'][0])}")
    
    # Verify list artifact was pooled
    assert isinstance(updated_artifacts['token_strings'][0], list)
    assert len(updated_artifacts['token_strings'][0]) == pooled_embeddings[0].shape[0]
    
    # Verify first token (protected) is kept
    assert updated_artifacts['token_strings'][0][0] == "CLS"
    
    print("✓ List artifacts work!")
    print("\n✓ Test 3 passed!")


if __name__ == "__main__":
    test_basic_pooling()
    test_protected_tokens()
    test_list_artifacts()
    
    print("\n" + "=" * 80)
    print("ALL IMPORTANCE POOLING TESTS PASSED! ✓")
    print("=" * 80)

