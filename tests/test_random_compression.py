"""
Tests for random pruning and pooling strategies.

Includes prints (for visibility) and assertions (for safety), matching other tests.
"""

import torch

from pylate.models.compression import (
    RandomPruningConfig,
    RandomPruningStrategy,
    RandomPoolingConfig,
    RandomPoolingStrategy,
)


def test_random_pruning():
    print("=" * 80)
    print("TEST 1: Random pruning")
    print("=" * 80)

    embeddings = [
        torch.randn(12, 16),
        torch.randn(8, 16),
    ]
    artifacts = {
        "input_ids": [torch.arange(12), torch.arange(8)],
    }

    cfg = RandomPruningConfig(
        keep_ratio=0.5,
        protected_tokens=1,
        min_tokens=5,
        seed=123,
    )
    strategy = RandomPruningStrategy(cfg)

    print(f"Original lengths: {[e.shape[0] for e in embeddings]}")
    pruned_emb, pruned_art = strategy.compress(embeddings, artifacts)
    print(f"Pruned lengths:   {[e.shape[0] for e in pruned_emb]}")

    for idx in range(len(embeddings)):
        assert pruned_emb[idx].shape[0] <= embeddings[idx].shape[0]
        assert pruned_emb[idx].shape[0] >= cfg.min_tokens
        assert pruned_emb[idx].shape[0] == pruned_art["input_ids"][idx].shape[0]
        # Protected token should remain first
        assert pruned_art["input_ids"][idx][0].item() == artifacts["input_ids"][idx][0].item()

    print("✓ Random pruning passed!")


def test_random_pooling():
    print("\n" + "=" * 80)
    print("TEST 2: Random pooling")
    print("=" * 80)

    embeddings = [torch.randn(15, 32)]
    artifacts = {"input_ids": [torch.arange(15)]}

    cfg = RandomPoolingConfig(
        keep_ratio=0.5,
        protected_tokens=2,
        min_tokens=6,
        seed=321,
    )
    strategy = RandomPoolingStrategy(cfg)

    print(f"Original length: {embeddings[0].shape[0]}")
    pooled_emb, pooled_art = strategy.compress(embeddings, artifacts)
    print(f"Pooled length:   {pooled_emb[0].shape[0]}")

    assert pooled_emb[0].shape[0] <= embeddings[0].shape[0]
    assert pooled_emb[0].shape[0] >= cfg.min_tokens
    assert pooled_emb[0].shape[0] == pooled_art["input_ids"][0].shape[0]
    # Protected tokens should be preserved at the front
    assert pooled_art["input_ids"][0][0].item() == 0
    assert pooled_art["input_ids"][0][1].item() == 1

    print("✓ Random pooling passed!")


if __name__ == "__main__":
    test_random_pruning()
    test_random_pooling()
    print("\nALL RANDOM COMPRESSION TESTS PASSED!")
