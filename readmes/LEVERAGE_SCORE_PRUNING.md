# Leverage Score Pruning for ColBERT

## Overview

This document describes the implementation of **Statistical Leverage Score Pruning** for token compression in the pylate ColBERT framework. This is a query-independent pruning method based on randomized numerical linear algebra.

## What are Leverage Scores?

Statistical leverage scores measure the **importance of each token (row) in the embedding matrix** for approximating the matrix's column space. For an embedding matrix K with SVD decomposition K = UΣVᵀ, the leverage score of token i is:

```
ℓᵢ = ||Uᵢ||²
```

where Uᵢ is the i-th row of the left singular vectors U.

**Intuition**: Tokens with high leverage scores are "influential" - they contribute more to the span of the embedding space and are harder to approximate by other tokens. Tokens with low leverage scores are redundant and can be safely pruned.

## Implementation Details

### Johnson-Lindenstrauss Random Projection

To make leverage score computation efficient, we use the **Johnson-Lindenstrauss (JL) lemma**:
- Instead of computing SVD on the full `n × d` embedding matrix (expensive)
- We project to a smaller `n × k` matrix where `k=64 << d` (cheap)
- Random projection approximately preserves distances and geometric structure
- Reduces complexity from O(n·d²) to O(n·k² + k³)

### Algorithm Steps

1. **Center embeddings** (optional): Subtract mean for numerical stability
2. **Random projection**: K_proj = K @ R where R is a random `d × k` matrix
3. **Gram matrix**: G = K_projᵀ @ K_proj (size `k × k`)
4. **SVD**: G = V @ diag(S) @ Vᵀ
5. **Compute U**: U = K_proj @ V @ S^(-1/2)
6. **Leverage scores**: ℓᵢ = ||Uᵢ||² for each token i
7. **Normalize** (optional): Z-score normalization for consistent thresholding
8. **Prune**: Remove tokens with lowest leverage scores

## Usage

### Basic Example

```python
from pylate.models.compression import (
    LeverageScorePruningConfig,
    LeverageScorePruningStrategy,
    CompressionConfig,
)

# Create configuration
config = LeverageScorePruningConfig(
    top_k=20,                    # Prune 20 tokens with lowest leverage scores
    protected_tokens=1,          # Keep first token (CLS)
    projection_dim=64,           # JL projection dimension
    center_embeddings=True,      # Center before computing scores
    normalize_scores=True,       # Z-score normalization
    track_pruned_tokens=False,   # Track which tokens were pruned
    show_progress_bar=True,      # Show progress during pruning
)

# Create strategy
strategy = LeverageScorePruningStrategy(config)

# Apply compression
pruned_embeddings, updated_artifacts = strategy.compress(
    embeddings=document_embeddings,
    artifacts={"input_ids": input_ids},  # Optional
)
```

### Integration with CompressionConfig

```python
# Combine with other strategies
compression_config = CompressionConfig(
    strategies=[
        LeverageScorePruningStrategy(LeverageScorePruningConfig(top_k=40)),
        PoolingStrategy(PoolingConfig(pool_factor=2)),
    ],
    description="Leverage pruning + pooling",
)

# Create compressor
compressor = compression_config.create_compressor()

# Apply all strategies in sequence
compressed_embeddings, artifacts = compressor.compress_parallel(
    embeddings=embeddings,
    artifacts={},
    batch_size=100,
    num_workers=8,
)
```

### Using in Experiments

```python
# In compression_experiment.py or similar
from pylate.models.compression import LeverageScorePruningConfig, LeverageScorePruningStrategy

# Add to experiment configs
for k in [5, 10, 20, 40, 80, 120, 160, 200]:
    config = CompressionConfig(
        strategies=[LeverageScorePruningStrategy(
            LeverageScorePruningConfig(
                top_k=k,
                protected_tokens=1,
                projection_dim=64,
            )
        )],
        description=f"Leverage score pruning k={k}",
    )
    configs.append(config)
```

## Configuration Options

### LeverageScorePruningConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | None | Number of tokens to prune (mutually exclusive with threshold) |
| `threshold` | float | None | Prune tokens with score < threshold |
| `protected_tokens` | int | 1 | Number of leading tokens to always keep |
| `projection_dim` | int | 64 | JL projection dimension (lower = faster, less accurate) |
| `center_embeddings` | bool | True | Center embeddings before computing scores |
| `normalize_scores` | bool | True | Apply z-score normalization |
| `track_pruned_tokens` | bool | False | Track which tokens were pruned |
| `show_progress_bar` | bool | False | Show progress during pruning |

## Advantages vs Other Methods

### vs Attention Pruning
- ✅ **Query-independent**: No need for forward pass or attention computation
- ✅ **Faster**: Only requires SVD on small projected matrix
- ✅ **Geometric**: Based on intrinsic structure of embeddings
- ❌ **Less adaptive**: Doesn't consider query-document interaction

### vs IDF Pruning
- ✅ **Context-aware**: Considers token relationships in embedding space
- ✅ **No corpus statistics needed**: Works on embeddings directly
- ✅ **Theoretically grounded**: Provable approximation guarantees
- ❌ **More compute**: SVD required per document

### vs Random Pruning
- ✅ **Principled**: Based on statistical importance
- ✅ **Better quality**: Preserves matrix structure
- ✅ **Reproducible**: Deterministic given embeddings

## Performance Considerations

- **Computational cost**: O(n·k² + k³) per document where n=tokens, k=projection_dim
- **Memory**: Minimal overhead, only stores k×k Gram matrix
- **Parallelization**: Fully parallelizable across documents
- **Typical projection_dim**: 32-128 (64 is a good default)

## Testing

Run the test script to verify the implementation:

```bash
cd pylate
python test_leverage_score_pruning.py
```

This tests:
1. Basic top_k pruning
2. Threshold-based pruning
3. Serialization/deserialization
4. Parallel compression

## References

- Drineas & Mahoney (2012): "Fast Approximation of Matrix Coherence and Statistical Leverage"
- Johnson-Lindenstrauss Lemma for dimensionality reduction
- Randomized Numerical Linear Algebra (RandNLA) literature

