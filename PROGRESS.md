# Progress: Straight-Through Estimator for Quantization-Aware Training

Branch `pylate-STE` (off `origin/main`). Adds a general straight-through estimator (STE)
so ColBERT can be trained with an *arbitrary* quantization transform applied to the
embeddings, using the existing loss functions unchanged.

## What landed

- **`pylate/models/quantization.py`**
  - `straight_through(x, q) = x + (q - x).detach()` — forward value is the quantized
    tensor, backward gradient is identity. Works with any (even non-differentiable)
    transform.
  - `Quantizer` base + `ScalarQuantizer` (uniform affine, n-bit, symmetric/asymmetric,
    fixed or per-vector range), `BinaryQuantizer` (sign), `IdentityQuantizer`, and a
    `QUANTIZERS` registry.
  - `StraightThroughEstimator(nn.Module)` — appended after `Dense`. It is a **permanent**
    module: saved/loaded with the model and active at inference, so it is the single
    source of truth for quantization (do **not** also pass `encode(precision=)`).
    Supports **asymmetric** query/document transforms.

- **Asymmetric routing.** `ColBERT.tokenize` injects an `is_query` flag into `features`;
  it rides through `ColBERTCollator` (`<column>_is_query`) and the trainer's feature
  regrouping into each per-sentence feature dict, so the module routes queries and
  documents to different transforms at both train and eval time.

- **ColBERT integration** (`models/colbert.py`): loader filter and the Dense-coercion /
  `embedding_size` guards updated so the STE module round-trips through
  `save_pretrained` / load.

- **Tests** (`tests/test_quantization.py`): STE gradient identity, quantizer grids,
  asymmetric routing, config round-trip, non-serializable-callable fallback, and an
  end-to-end asymmetric-QAT training + save/reload test.

- **Config-driven runner** (`examples/train/qat_run.py` + `examples/train/configs/`):
  enabling/disabling the `quantization` block gives a QAT run or an iso (full-precision)
  baseline from the same script. wandb optional (`--with wandb`; creds in `~/.netrc`).

## Result (10k steps)

MiniLM-L6 base + random projection, `sentence-transformers/msmarco-bm25` triplets,
batch 32, lr 1e-5, bf16, eval = triplet accuracy on 1k held-out. wandb project
`robro612/pylate-ste`.

| | fp32 baseline | QAT (int8 query / 1-bit document) | gap |
|---|---|---|---|
| final (step 10k) | 0.834 | 0.811 | 2.3 |
| peak | 0.839 | 0.817 | 2.2 |

Gap stays in a tight ~2–2.5 pt band across all matched checkpoints — small cost for
binarized document embeddings. Saved models:
`output/{qat-int8-binary,baseline-fp32}-10k/final`.

## How to run

```bash
cd /exp/rjha/pylate-STE
# QAT run (int8 query / binary doc):
srunl40s uv run --with wandb python -u examples/train/qat_run.py \
  --config examples/train/configs/qat_int8_binary_10k.yaml
# Iso baseline (full precision, identical otherwise):
srunl40s uv run --with wandb python -u examples/train/qat_run.py \
  --config examples/train/configs/baseline_fp32_10k.yaml
# Tests:
srunl40s uv run --extra dev python -m pytest tests/test_quantization.py \
  pylate/models/quantization.py -q
```

## Next steps / open questions

- Numbers are modest because training is MiniLM + random projection for ~0.8 epoch; the
  *relative* QAT cost is the signal. Starting QAT from an already-trained ColBERT would
  likely shrink the gap.
- Consider int4 documents, learnable per-dim scales, or a global (calibrated) range
  instead of per-vector for closer parity with the serving index quantizer.
