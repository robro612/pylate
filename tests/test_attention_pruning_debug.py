"""
Debug test for attention pruning to understand what's happening.
"""

import sys
import os

import torch

from pylate.models.compression import (
    AttentionPruningConfig,
    AttentionPruningStrategy,
)


def test_attention_pruning_debug():
    """Test attention pruning with debug output."""
    print("=" * 80)
    print("ATTENTION PRUNING DEBUG TEST")
    print("=" * 80)
    
    # Create sample embeddings
    embeddings = [
        torch.randn(10, 128),
        torch.randn(8, 128),
    ]
    
    # Create sample attention scores (higher = more important)
    attention_scores = [
        torch.tensor([0.9, 0.8, 0.3, 0.5, 0.2, 0.7, 0.4, 0.6, 0.1, 0.05]),
        torch.tensor([0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15]),
    ]
    
    artifacts = {
        "attention_scores": attention_scores,
        "input_ids": [
            torch.arange(10),
            torch.arange(8),
        ],
    }
    
    # Create config with debug enabled
    config = AttentionPruningConfig(
        top_k=3,  # Remove 3 tokens with LOWEST attention
        protected_tokens=1,
        normalize_scores=False,
    )
    
    # Create strategy with debug=True
    strategy = AttentionPruningStrategy(config, debug=True)
    
    print(f"\nOriginal lengths: {[e.shape[0] for e in embeddings]}")
    print(f"Config: top_k={config.top_k}, protected_tokens={config.protected_tokens}")
    print(f"\nAttention scores:")
    for i, scores in enumerate(attention_scores):
        print(f"  Doc {i}: {scores.tolist()}")
    
    # Apply compression
    pruned_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"\nPruned lengths: {[e.shape[0] for e in pruned_embeddings]}")
    print(f"Pruned input_ids: {[ids.tolist() for ids in updated_artifacts['input_ids']]}")
    
    # Verify
    assert pruned_embeddings[0].shape[0] == 10 - 3  # 10 - 3 removed = 7
    assert pruned_embeddings[1].shape[0] == 8 - 3   # 8 - 3 removed = 5
    
    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_attention_pruning_debug()

