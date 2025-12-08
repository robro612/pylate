# Unnormalized Embeddings for Importance-Based Compression

## Summary

Modified the compression experiment to use **unnormalized embeddings** for importance-based compression strategies, then normalize after compression. This is critical for L2 norm-based importance scoring to work correctly.

## The Problem

When embeddings are normalized (L2 norm = 1.0 for all tokens):
- **All tokens have the same L2 norm** (~1.0)
- The `use_norm` component in importance scoring provides **NO discrimination**
- Token selection becomes essentially random (unless IDF or other signals are used)

## The Solution

### 1. Encode with `normalize_embeddings=False`

**File**: `pylate/experiments/compression/compression_experiment.py`

```python
documents_embeddings, artifacts = model.encode(
    sentences=[document["text"] for document in documents],
    batch_size=args.batch_size,
    is_query=False,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=False,  # ← Keep unnormalized for importance scoring
    return_extra_artifacts={"input_ids": True, "attention_scores": True},
)
```

### 2. Normalize AFTER Compression

```python
# Compress with unnormalized embeddings
compressed_embeddings, _ = compressor.compress_parallel(
    embeddings=documents_embeddings,  # Unnormalized
    artifacts=artifacts,
    batch_size=args.batch_size,
    num_workers=8,
    show_progress=True,
)

# Normalize embeddings AFTER compression
import torch.nn.functional as F
compressed_embeddings = [
    F.normalize(emb, p=2, dim=-1) for emb in compressed_embeddings
]
```

### 3. Also Normalize Baseline (No Compression)

```python
if config is None:
    # Baseline: normalize the unnormalized embeddings
    import torch.nn.functional as F
    compressed_embeddings = [
        F.normalize(emb, p=2, dim=-1) for emb in documents_embeddings
    ]
```

## Why This Works

### Unnormalized Embeddings Provide Discrimination

Example from `test_unnormalized_importance.py`:

```
Unnormalized Embeddings:
Token 1: [3.0, 4.0] -> L2 norm = 5.000  (important)
Token 2: [1.5, 2.0] -> L2 norm = 2.500  (medium)
Token 3: [0.3, 0.4] -> L2 norm = 0.500  (unimportant)

Normalized Embeddings:
Token 1: [0.6, 0.8] -> L2 norm = 1.000
Token 2: [0.6, 0.8] -> L2 norm = 1.000
Token 3: [0.6, 0.8] -> L2 norm = 1.000

Discrimination Power:
Unnormalized: 10.0x discrimination (0.5 to 5.0)
Normalized:   1.0x discrimination (all 1.0)
```

### Importance Scoring Formula

```python
scores = torch.zeros(n_tokens, device=device)

if self.config.use_norm:
    norms = torch.norm(doc_embeddings, dim=-1)
    scores = scores + self.config.norm_weight * norms  # ← Now varies!

if self.config.use_idf and idf_vec is not None:
    scores = scores + self.config.idf_weight * idf_vec

if self.config.use_token_weights and token_weight_vec is not None:
    scores = scores + self.config.token_weights_weight * token_weight_vec
```

With unnormalized embeddings:
- ✅ **L2 norm component**: Varies significantly across tokens (provides discrimination)
- ✅ **IDF component**: Still works as before
- ✅ **Token weights component**: Still works as before

## Impact on Strategies

### Affected Strategies (use importance scoring)
1. **ImportancePruningStrategy** - Now uses meaningful L2 norms
2. **ImportancePoolingStrategy** - Now uses meaningful L2 norms
3. **HybridImportanceClusteringPoolingStrategy** - Now uses meaningful L2 norms

### Unaffected Strategies (don't use L2 norm)
- **AttentionPruningStrategy** - Uses attention scores
- **LeverageScorePruningStrategy** - Uses leverage scores
- **IDFPruningStrategy** - Uses IDF/TF-IDF scores
- **RandomPruningStrategy** - Random selection
- **RandomPoolingStrategy** - Random selection
- **PoolingStrategy** - Clustering-based

## Files Modified

1. **`pylate/experiments/compression/compression_experiment.py`**
   - Line 923: Added `normalize_embeddings=False` to document encoding
   - Lines 997-1002: Normalize baseline embeddings
   - Lines 1015-1019: Normalize compressed embeddings

## Testing

Created `test_unnormalized_importance.py` to demonstrate:
- ✅ Unnormalized embeddings provide 10x discrimination power
- ✅ Normalized embeddings provide no discrimination (all norms = 1.0)

## Benefits

1. **L2 norm now meaningful**: Tokens with larger embeddings are considered more important
2. **Better token selection**: Importance-based strategies can now distinguish important from unimportant tokens
3. **Backward compatible**: Queries are still normalized (as before)
4. **Final embeddings normalized**: After compression, embeddings are normalized for retrieval (as expected)

## Important Notes

- **Queries remain normalized**: Query encoding is unchanged (still uses `normalize_embeddings=True` by default)
- **Final embeddings are normalized**: After compression, all document embeddings are normalized for retrieval
- **Only affects importance-based strategies**: Other strategies are unaffected by this change

