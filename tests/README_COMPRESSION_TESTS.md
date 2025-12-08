# Compression Strategy Tests

This directory contains test files for various compression strategies in pylate.

## Test Files

### Attention Pruning
- **`test_attention_pruning_debug.py`**: Debug test for attention pruning with detailed output
  - Shows attention scores and pruning decisions
  - Verifies basic pruning functionality
  - Run: `python tests/test_attention_pruning_debug.py`

### Leverage Score Pruning
- **`test_leverage_score_fixes.py`**: Comprehensive tests for the fixed LeverageScorePruningStrategy
  - **Test 1**: `min_tokens` constraint prevents over-pruning
  - **Test 2**: Artifact alignment (cached masks, no recomputation)
  - **Test 3**: List artifacts handling (non-tensor artifacts)
  - **Test 4**: Config validation (catches invalid configs)
  - Run: `python tests/test_leverage_score_fixes.py`

### Importance Pruning
- **`test_importance_pruning.py`**: Tests for ImportancePruningStrategy
  - **Test 1**: Basic pruning functionality
  - **Test 2**: IDF integration
  - **Test 3**: `min_tokens` constraint
  - Run: `python tests/test_importance_pruning.py`

- **`test_importance_pruning_serialization.py`**: Serialization tests
  - Strategy serialization/deserialization
  - CompressionConfig integration
  - Run: `python tests/test_importance_pruning_serialization.py`

### Importance Pooling
- **`test_importance_pooling.py`**: Tests for ImportancePoolingStrategy
  - **Test 1**: Basic pooling functionality
  - **Test 2**: Protected tokens preservation
  - **Test 3**: List artifacts handling
  - Run: `python tests/test_importance_pooling.py`

- **`test_importance_pooling_serialization.py`**: Serialization tests
  - Strategy serialization/deserialization
  - CompressionConfig integration
  - Run: `python tests/test_importance_pooling_serialization.py`

## Running Tests

### Individual Test Files
```bash
conda activate pl
cd pylate
python tests/test_importance_pooling.py
python tests/test_leverage_score_fixes.py
python tests/test_importance_pruning.py
```

### All Tests (if pytest is available)
```bash
conda activate pl
cd pylate
pytest tests/test_importance_*.py -v
pytest tests/test_leverage_score_*.py -v
pytest tests/test_attention_*.py -v
```

## Test Coverage

### What's Tested
✅ Basic compression functionality (pruning/pooling)
✅ Artifact alignment (embeddings and artifacts stay synchronized)
✅ Protected tokens preservation
✅ `min_tokens` constraint enforcement
✅ Both tensor and list artifacts
✅ Serialization/deserialization
✅ Config validation
✅ IDF integration (for importance-based strategies)

### Key Fixes Verified
✅ **Leverage Score**: Masks cached and reused (no recomputation for artifacts)
✅ **Leverage Score**: Both tensor and list artifacts handled correctly
✅ **Leverage Score**: `min_tokens` constraint prevents over-pruning
✅ **All Strategies**: Artifacts stay aligned with compressed embeddings

## Test Output Examples

### Success Output
```
================================================================================
TEST 1: Basic pooling
================================================================================

Original lengths: [20, 15]
Config: keep_ratio=0.5, min_tokens=8
Pooled lengths:   [10, 8]
Pooled input_ids: [10, 8]
✓ Basic pooling works!

✓ Test 1 passed!
```

### All Tests Passed
```
================================================================================
ALL IMPORTANCE POOLING TESTS PASSED! ✓
================================================================================
```

## Notes

- All test files are standalone and can be run independently
- Tests use synthetic data (random embeddings) for reproducibility
- Debug output is included to help understand what each strategy does
- Tests verify both correctness and edge cases (e.g., over-pruning prevention)

## Future Test Ideas

- [ ] Performance benchmarks (compression ratio vs quality)
- [ ] Integration tests with real ColBERT models
- [ ] Stress tests with very long documents
- [ ] Comparison tests between different strategies
- [ ] Tests with real IDF scores from actual corpora

