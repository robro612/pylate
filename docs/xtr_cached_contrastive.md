# XTR Support in CachedContrastive

## Background

`CachedContrastive` uses gradient caching (GradCache) to decouple the effective contrastive batch size from GPU memory. It embeds inputs in small chunks without grad, computes the full-batch contrastive loss, caches embedding gradients, then replays the forward passes with grad to backprop into the model.

Scoring is handled by `ScopedBatchScores` from the loss classes. For ColBERT mode, query, document-batch, and N-way chunking are supported. For XTR mode, query chunking is supported while document dimensions are scored globally.

## Current approach (implemented)

Use `ScopedBatchScores(mode="xtr")` with optional query chunking and full global document scoring in each query chunk.

### Memory analysis

The dominant tensor is the token-level score matrix: `(mbsz, Qt, batch*N*Dt)`.

- `mbsz` (mini_batch_size) is the tuning knob — halving it halves peak memory.
- With mbsz=32, Qt=32, batch\*N=128, Dt=180: ~90MB in fp32. Manageable.
- Having more in-batch docs (larger N) just means using a proportionally smaller `mini_batch_size` to keep memory constant.

### Label adjustment

For XTR, the score matrix is `(mbsz, batch*N)` — documents are interleaved per query. The positive for query `i` is at column `i*N`, not `i`. Labels must be adjusted accordingly.

## Approach 2: Two-pass streaming top-k (future work)

If Approach 1 hits OOM at very large batch sizes, the global top-k can be computed as a streaming reduce over document chunks in two passes:

**Pass 1 — find threshold**: Stream through doc chunks, maintaining a running `(mbsz, Qt, k)` top-k buffer per query token. After all chunks, the k-th value is the threshold.

**Pass 2 — masked MaxSim**: Stream through doc chunks again, zero out token scores below threshold, compute MaxSim per doc, accumulate.

This reduces peak memory from `O(mbsz * Qt * total_docs * Dt)` to `O(mbsz * Qt * max(chunk_docs * Dt, k))` at the cost of 2x compute (two passes). Threshold-based masking instead of exact index masking is a minor approximation — float score ties at the boundary are rare and harmless for training.

Could be implemented as a standalone `xtr_scores_chunked` function, callable from `CachedContrastive` as a drop-in replacement when memory is tight.
