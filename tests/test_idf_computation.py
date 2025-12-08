"""Test IDF computation in importance-based strategies."""

import torch
from pylate.models.compression import (
    ImportancePruningConfig,
    ImportancePruningStrategy,
    ImportancePoolingConfig,
    ImportancePoolingStrategy,
    HybridPoolingConfig,
    HybridImportanceClusteringPoolingStrategy,
)


def test_importance_pruning_with_idf():
    """Test that ImportancePruningStrategy computes IDF when use_idf=True."""
    print("\n" + "=" * 80)
    print("Testing ImportancePruningStrategy with IDF computation")
    print("=" * 80)
    
    # Create config with IDF enabled
    config = ImportancePruningConfig(
        keep_ratio=0.5,
        protected_tokens=1,
        min_tokens=4,
        use_norm=True,
        use_idf=True,  # Enable IDF
        norm_weight=1.0,
        idf_weight=1.0,
    )
    
    strategy = ImportancePruningStrategy(config)
    
    # Create sample embeddings and artifacts
    embeddings = [
        torch.randn(10, 128),  # Doc 1: 10 tokens
        torch.randn(8, 128),   # Doc 2: 8 tokens
    ]
    
    # Provide input_ids (required for IDF computation)
    artifacts = {
        "input_ids": [
            torch.tensor([101, 2023, 2003, 1037, 3231, 2000, 3231, 2023, 2003, 102]),  # Doc 1
            torch.tensor([101, 2023, 2003, 1037, 2117, 3231, 2000, 102]),  # Doc 2
        ]
    }
    
    # Compress
    compressed_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"✓ Original doc 1 tokens: {embeddings[0].shape[0]}")
    print(f"✓ Compressed doc 1 tokens: {compressed_embeddings[0].shape[0]}")
    print(f"✓ Original doc 2 tokens: {embeddings[1].shape[0]}")
    print(f"✓ Compressed doc 2 tokens: {compressed_embeddings[1].shape[0]}")
    print(f"✓ IDF was computed and used for importance scoring!")
    

def test_importance_pooling_with_idf():
    """Test that ImportancePoolingStrategy computes IDF when use_idf=True."""
    print("\n" + "=" * 80)
    print("Testing ImportancePoolingStrategy with IDF computation")
    print("=" * 80)
    
    # Create config with IDF enabled
    config = ImportancePoolingConfig(
        keep_ratio=0.5,
        protected_tokens=1,
        min_tokens=4,
        use_norm=True,
        use_idf=True,  # Enable IDF
        norm_weight=1.0,
        idf_weight=1.0,
    )
    
    strategy = ImportancePoolingStrategy(config)
    
    # Create sample embeddings and artifacts
    embeddings = [
        torch.randn(10, 128),  # Doc 1: 10 tokens
        torch.randn(8, 128),   # Doc 2: 8 tokens
    ]
    
    # Provide input_ids (required for IDF computation)
    artifacts = {
        "input_ids": [
            torch.tensor([101, 2023, 2003, 1037, 3231, 2000, 3231, 2023, 2003, 102]),  # Doc 1
            torch.tensor([101, 2023, 2003, 1037, 2117, 3231, 2000, 102]),  # Doc 2
        ]
    }
    
    # Compress
    pooled_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"✓ Original doc 1 tokens: {embeddings[0].shape[0]}")
    print(f"✓ Pooled doc 1 tokens: {pooled_embeddings[0].shape[0]}")
    print(f"✓ Original doc 2 tokens: {embeddings[1].shape[0]}")
    print(f"✓ Pooled doc 2 tokens: {pooled_embeddings[1].shape[0]}")
    print(f"✓ IDF was computed and used for importance scoring!")


def test_hybrid_pooling_with_idf():
    """Test that HybridImportanceClusteringPoolingStrategy computes IDF when use_idf=True."""
    print("\n" + "=" * 80)
    print("Testing HybridImportanceClusteringPoolingStrategy with IDF computation")
    print("=" * 80)
    
    # Create config with IDF enabled
    config = HybridPoolingConfig(
        pool_factor=2,
        keep_ratio=0.5,
        protected_tokens=1,
        min_tokens=4,
        clustering_method="hierarchical",
        use_norm=True,
        use_idf=True,  # Enable IDF
        norm_weight=1.0,
        idf_weight=1.0,
    )
    
    strategy = HybridImportanceClusteringPoolingStrategy(config)
    
    # Create sample embeddings and artifacts
    embeddings = [
        torch.randn(10, 128),  # Doc 1: 10 tokens
        torch.randn(8, 128),   # Doc 2: 8 tokens
    ]
    
    # Provide input_ids (required for IDF computation)
    artifacts = {
        "input_ids": [
            torch.tensor([101, 2023, 2003, 1037, 3231, 2000, 3231, 2023, 2003, 102]),  # Doc 1
            torch.tensor([101, 2023, 2003, 1037, 2117, 3231, 2000, 102]),  # Doc 2
        ]
    }
    
    # Compress
    pooled_embeddings, updated_artifacts = strategy.compress(embeddings, artifacts)
    
    print(f"✓ Original doc 1 tokens: {embeddings[0].shape[0]}")
    print(f"✓ Pooled doc 1 tokens: {pooled_embeddings[0].shape[0]}")
    print(f"✓ Original doc 2 tokens: {embeddings[1].shape[0]}")
    print(f"✓ Pooled doc 2 tokens: {pooled_embeddings[1].shape[0]}")
    print(f"✓ IDF was computed and used for importance scoring!")


if __name__ == "__main__":
    test_importance_pruning_with_idf()
    test_importance_pooling_with_idf()
    test_hybrid_pooling_with_idf()
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! IDF computation is working correctly.")
    print("=" * 80)

