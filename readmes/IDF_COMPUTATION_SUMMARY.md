# IDF Computation in Importance-Based Compression Strategies

## Summary

Added automatic document-wise IDF computation to three importance-based compression strategies:
1. `ImportancePruningStrategy`
2. `ImportancePoolingStrategy`
3. `HybridImportanceClusteringPoolingStrategy`

## What Changed

### Before
When `use_idf=True` was set in the config, the strategies would **only use IDF if it was already present** in the `artifacts` dictionary. If `artifacts["idf"]` was `None` or missing, the IDF component was silently ignored.

### After
When `use_idf=True` is set in the config:
1. The strategy first checks if `artifacts["idf"]` exists
2. If **not present** and `artifacts["input_ids"]` is available, it **automatically computes document-wise IDF scores**
3. The computed IDF scores are then used for importance scoring

## Implementation Details

### New Method: `_compute_document_idf()`

Added to all three strategy classes:

```python
def _compute_document_idf(
    self,
    input_ids: list[torch.Tensor],
    show_progress: bool = False,
) -> list[torch.Tensor]:
    """
    Compute document-wise IDF scores for each token.
    
    Parameters
    ----------
    input_ids
        List of input_id tensors (one per document)
    show_progress
        Whether to show progress bar
    
    Returns
    -------
    list[torch.Tensor]
        List of IDF score tensors (one per document), same shape as input_ids
    """
    # Convert tensors to lists for TokenTFIDFStats
    tokenized_docs = [doc_input_ids.cpu().tolist() for doc_input_ids in input_ids]
    
    # Compute TF-IDF statistics
    stats = TokenTFIDFStats(num_docs=len(tokenized_docs))
    stats.fit(tokenized_docs, show_progress=show_progress)
    
    # Extract IDF scores for each document
    idf_scores = []
    for doc_idx, doc_tokens in enumerate(tokenized_docs):
        doc_idf = torch.tensor(
            [stats.get_idf(token_id) for token_id in doc_tokens],
            dtype=torch.float32,
        )
        idf_scores.append(doc_idf)
    
    return idf_scores
```

### Updated `compress()` Method

Added IDF computation logic at the beginning of each `compress()` method:

```python
# Compute IDF if needed and not provided
if self.config.use_idf and idf_docs is None:
    input_ids = artifacts.get("input_ids", None)
    if input_ids is not None:
        idf_docs = self._compute_document_idf(input_ids, show_progress=False)
```

## Files Modified

1. **`pylate/pylate/models/compression/importance_pruning.py`**
   - Added `_compute_document_idf()` method
   - Updated `compress()` to compute IDF when needed
   - Added import for `TokenTFIDFStats`

2. **`pylate/pylate/models/compression/importance_pooling.py`**
   - Added `_compute_document_idf()` method
   - Updated `compress()` to compute IDF when needed
   - Added import for `TokenTFIDFStats`

3. **`pylate/pylate/models/compression/hybrid_importance_pooling.py`**
   - Added `_compute_document_idf()` method
   - Updated `compress()` to compute IDF when needed
   - Added import for `TokenTFIDFStats`

## Usage Example

```python
from pylate.models import ImportancePoolingConfig, ImportancePoolingStrategy

# Create config with IDF enabled
config = ImportancePoolingConfig(
    keep_ratio=0.5,
    protected_tokens=1,
    min_tokens=8,
    use_norm=True,
    use_idf=True,  # Enable IDF - will be computed automatically!
    norm_weight=1.0,
    idf_weight=1.0,
)

strategy = ImportancePoolingStrategy(config)

# Compress with only input_ids in artifacts
# IDF will be computed automatically from input_ids
compressed, _ = strategy.compress(
    embeddings=embeddings,
    artifacts={"input_ids": input_ids}  # No need to pre-compute IDF!
)
```

## Benefits

1. **Automatic**: No need to manually compute IDF before compression
2. **Efficient**: IDF is only computed when `use_idf=True` and not already provided
3. **Consistent**: Uses the same IDF computation method as `IDFPruningStrategy`
4. **Backward Compatible**: If `artifacts["idf"]` is already provided, it will be used directly

## Testing

Created `test_idf_computation.py` to verify:
- ✅ ImportancePruningStrategy computes and uses IDF correctly
- ✅ ImportancePoolingStrategy computes and uses IDF correctly
- ✅ HybridImportanceClusteringPoolingStrategy computes and uses IDF correctly

All tests pass successfully!

