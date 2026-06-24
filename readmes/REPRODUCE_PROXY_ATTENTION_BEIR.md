# Reproducing & Extending Proxy-Attention (Attention-Guided Clustering) on BEIR

This guide documents how to reproduce the **proxy-attention** results from scratch and how
to extend them to new vector budgets (e.g. **8 or 16 vectors per document**), which is what
most follow-up requests ask for. The method is implemented as the **`ProxyAttentionColBERT`**
model: learnable proxy query tokens compute attention saliency over document tokens, the
top-`m` most salient tokens are selected, and (optionally) nearby tokens are pooled into those
centroids — i.e. *attention-guided clustering*.

> **TL;DR** — Everything needed (model, loss, training entrypoint, configs, BEIR eval harness,
> tests) is committed. The trained checkpoints are *not* shipped, so each vector budget requires
> training one model. A different budget (8 vs 16) = a separately trained model. The existing
> released checkpoints were trained with the **Hydra** trainer (Section 4) on a **single H100**.

---

## 1. What this method is (and is not)

There are **two distinct compression families** in this repo. Don't confuse them:

| | Trained proxy-attention | Post-hoc attention pooling |
|---|---|---|
| Class / strategy | `models.ProxyAttentionColBERT` | `compression.AttentionPoolingStrategy` |
| Requires training? | **Yes** — learned proxy tokens | No — uses a frozen model's attention |
| Vector budget knob | `num_select_tokens` (train time) | `keep_ratio` (eval time) |
| Eval harness | `examples/evaluation/beir_dataset.py` | `experiments/compression/compression_eval.py` |

This guide covers the **trained** path (`ProxyAttentionColBERT`), since that is what "proxy-attn"
refers to and what requires a new model per vector budget. The post-hoc `compression_eval.py`
pipeline is a separate set of *training-free* baselines (random / attention / IDF / leverage
pruning & pooling) applied on top of a stock `lightonai/GTE-ModernColBERT-v1`.

---

## 2. Files involved (all committed to git)

**Method**
- `pylate/models/ProxyAttentionColBERT.py` — the model. Key knobs: `num_proxy_tokens` (nψ),
  `num_select_tokens` (m, the output vector count), `proxy_tau`, `use_cluster_pooling`,
  `cluster_centroid_weight`, `use_attn_weight_cluster_pooling`. Has custom `save()` / `load()`.
- `pylate/losses/proxy_attention_distillation.py` — `ProxyAttentionDistillation` (KL-divergence
  knowledge-distillation loss).

**Training (the path that produced the released checkpoints)**
- `examples/train/gte_modern_colbert_hydra.py` — Hydra/OmegaConf training entrypoint.
- `scripts/train_variants.sh` — SLURM-array launcher; the proxy-attention models are array
  indices 1 / 4 / 7 (P32-S32 / P24-S24 / P16-S16).
- `conf/gte_modern_colbert.yaml` — base training config (batch size, steps, lr, dataset, …).
- `conf/model/proxy_attention.yaml` — the proxy-attention model variant config.
- `examples/train/PROXY_ATTENTION_COLBERT.md` — math + code walkthrough of the method.
- `examples/train/proxy_attention_colbert.py` — an *alternative* standalone argparse trainer.
  **It was not used for the released checkpoints** (it schedules by `--epochs` rather than
  `max_steps` and defaults to a different batch size). Prefer the Hydra path for faithful repro.

**Evaluation**
- `examples/evaluation/beir_dataset.py` — **full BEIR test eval** with a PLAID index
  (`--model_type proxy_attention`). Already loads proxy models correctly via `.load()`
  (committed fix, see Section 5).
- `pylate/evaluation/beir.py` — BEIR dataset loader used by the above.
- `evaluation.NanoBEIREvaluator` — small-subset eval used *during training* for monitoring;
  **not** a substitute for full BEIR test numbers.

**Tests** — `tests/proxy_attention/` (model, loss, attention-capture, integration).

**Not committed (expected):** trained checkpoints under `output/proxy_attention-*/`. These are
weights, not source; reproduction re-creates them.

---

## 3. Environment

```bash
# from the repo root
uv pip install -e .          # or: pip install -e .
```

The base model and KD dataset are pulled from the HuggingFace Hub automatically:
- Base encoder: `Alibaba-NLP/gte-modernbert-base` (what the released checkpoints used).
- KD training data: `lightonai/ms-marco-en-bge-gemma` (MS MARCO with BGE-Gemma teacher scores).

Hardware: the released checkpoints were trained on a **single H100** (`--gres=gpu:h100:1`,
torch `2.9.0+cu128`). A100 / L40S work too. **V100 is incompatible with the current cu13 torch
build** (Volta sm_70 is dropped), so use it only for CPU-side work.

---

## 4. Train the models (one per vector budget)

The output vector count **m** is `num_select_tokens` and is fixed at *training* time. To get an
8-vector and a 16-vector model, train two models. **Exact command used for the released
16-vector model** (`scripts/train_variants.sh`, array index 7):

```bash
python examples/train/gte_modern_colbert_hydra.py \
  --config-name gte_modern_colbert \
  model=proxy_attention \
  model.variant_args.num_proxy_tokens=16 \
  model.variant_args.num_select_tokens=16 \
  compile=false
```

For **8 vectors**, change both overrides to `8`:

```bash
python examples/train/gte_modern_colbert_hydra.py \
  --config-name gte_modern_colbert \
  model=proxy_attention \
  model.variant_args.num_proxy_tokens=8 \
  model.variant_args.num_select_tokens=8 \
  compile=false
```

What the configs supply (resolved values, verified against a released checkpoint's saved
`config.yaml`):

| Source | Setting | Value |
|---|---|---|
| `conf/gte_modern_colbert.yaml` | `batch_size` | **20** |
| | `n_ways` | 16 |
| | `lr` | 3e-5 |
| | `max_steps` | **10000** |
| | `warmup_ratio` | 0.0 |
| | `eval_steps` / `save_steps` / `logging_steps` | 250 / 5000 / 10 |
| | `dtype` | bf16 |
| | `model.document_length` / `query_length` | 300 / 32 |
| | `dataset_path` | `lightonai/ms-marco-en-bge-gemma` |
| `conf/model/proxy_attention.yaml` | `use_cluster_pooling` | true |
| | `proxy_tau` | 1.0 |
| | `num_proxy_tokens` / `num_select_tokens` | 32 / 32 (override per run) |

Notes:
- `compile=false` is set explicitly for proxy-attention runs (the base config has `compile=true`,
  but `torch.compile` doesn't play well with the eager-attention capture this model needs).
- The released checkpoints also used `use_attn_weight_cluster_pooling=true` (the model default):
  saliency scores weight the pooling. It isn't set in the config, so leave the default in place.
- To launch all variants as the original SLURM array (single H100 each):
  `sbatch scripts/train_variants.sh` (proxy indices are 1/4/7; add an `8`-token entry to the
  `configs` array if you want that budget in the sweep).
- Single-GPU interactive run: drop the SLURM wrapper and run the `python ...` command directly
  on an H100/A100/L40S node.
- Mid-training NanoBEIR scores are logged but are *not* the reported BEIR test numbers.
- The trained model is written under the run's output dir as `.../final` (custom `save()` writes
  `proxy_embeddings/` and records the proxy params in `config_sentence_transformers.json`).

---

## 5. Evaluate on BEIR

The loading is already correct in the committed code. `examples/evaluation/beir_dataset.py`
loads `proxy_attention` models via `ProxyAttentionColBERT.load(...)`, which restores the trained
`num_select_tokens` and the learned proxy embeddings from the checkpoint.

> **Background (why `.load()` matters):** the plain `ProxyAttentionColBERT(...)` constructor does
> **not** read the saved proxy config — it would default to `num_select_tokens=32` and randomly
> re-initialize the proxy embeddings, silently evaluating the wrong model. This was fixed in the
> eval script (commit `342a9a7`) so the constructor is no longer used for this model type. If you
> write your own eval, load with `ProxyAttentionColBERT.load(path)`, not the bare constructor.

```bash
python examples/evaluation/beir_dataset.py \
    --model_type proxy_attention \
    --model_name_or_path output/<your-16tok-run>/final \
    --index_type plaid \
    --output_dir evaluation_results/proxy_attention-S16 \
    --save_runfile \
    --dataset_name nfcorpus scifact fiqa trec-covid scidocs arguana \
                   webis-touche2020 quora nq hotpotqa fever \
                   climate-fever dbpedia-entity
```

This loads each BEIR dataset, encodes docs and queries, builds a PLAID index, retrieves, and
writes `evaluation_results/.../overall_results.jsonl` plus per-dataset `evaluation_results.json`
(and `run.json` runfiles with `--save_runfile`). Reported metrics: `map, ndcg@10, ndcg@100,
recall@10, recall@100`. Repeat with the 8-vector checkpoint.

**Sanity check the budget loaded:** the script prints `avg_tokens_per_document`. For a correctly
loaded model it should be **≈ `num_select_tokens`** (8 or 16) — not ~300, and not 32. (Verified:
the P16-S16 checkpoint encodes exactly 16 vectors/doc through this path.)

Per-dataset query lengths are encoded in the `query_len` dict at the top of the script.

---

## 6. Extending the method

- **Other vector budgets** — set `model.variant_args.num_select_tokens` (and `num_proxy_tokens`)
  to any value (4, 8, 16, 24, 32, …). One model per budget.
- **Pooling variants** — `model.variant_args.use_cluster_pooling=false` (pure top-k selection),
  plus `cluster_centroid_weight` and `use_attn_weight_cluster_pooling` (constructor args; add
  them to `conf/model/proxy_attention.yaml` to sweep).
- **Saliency temperature** — `model.variant_args.proxy_tau` sharpens/softens the proxy→token
  attention softmax.
- **Different base encoder** — `model.name=<any ColBERT-compatible encoder>`. Attention-capture
  support lives in `_get_last_layer` (ModernBERT / BERT / RoBERTa styles handled; add a branch
  for others).
- **New datasets** — add the dataset to `--dataset_name` in `beir_dataset.py`; set its query
  length in the `query_len` dict. `cqadupstack/*` subsets are handled specially (auto-downloaded).

---

## 7. Gotchas & caveats checklist

- [ ] **Use the Hydra trainer for faithful repro** (Section 4), not `proxy_attention_colbert.py`
      — the released checkpoints used `max_steps=10000` and `batch_size=20`, not epochs/bs=24.
- [ ] **Checkpoints are not in git.** Train first; nothing is reproducible from committed weights.
- [ ] **8 ≠ 16 is two training runs**, not an eval-time flag. The budget is baked in at train time.
- [ ] **NanoBEIR ≠ BEIR.** The in-training evaluator uses small subsets; report numbers from
      `beir_dataset.py`.
- [ ] **`.load()`, not the constructor**, if you write a custom eval. The committed eval script
      already does this; the constructor silently yields a random 32-vector model otherwise.
- [ ] **`compile=false` for proxy runs** — `torch.compile` breaks the eager-attention capture.
- [ ] **GPU:** trained on a single H100; A100/L40S fine; V100 is incompatible with the cu13 torch
      build (use CPU there).
- [ ] **Working-tree drift.** As of this writing several eval-side files have *uncommitted local
      modifications* (`experiments/compression/compression_eval*.py`, its `conf/`, and
      `pylate/models/compression/*.py`). The proxy training/eval path is unaffected, but if you
      hand this off, commit or stash those so the recipient isn't chasing local-only behavior.

---

## 8. One-screen recipe

```bash
# 0. install
uv pip install -e .

# 1. train (two budgets, single H100/A100/L40S each)
python examples/train/gte_modern_colbert_hydra.py --config-name gte_modern_colbert \
    model=proxy_attention model.variant_args.num_proxy_tokens=8  \
    model.variant_args.num_select_tokens=8  compile=false
python examples/train/gte_modern_colbert_hydra.py --config-name gte_modern_colbert \
    model=proxy_attention model.variant_args.num_proxy_tokens=16 \
    model.variant_args.num_select_tokens=16 compile=false

# 2. eval on BEIR (loading fix already committed)
python examples/evaluation/beir_dataset.py --model_type proxy_attention \
    --model_name_or_path output/<your-8tok-run>/final \
    --index_type plaid --save_runfile --output_dir evaluation_results/S8 \
    --dataset_name nfcorpus scifact fiqa trec-covid scidocs arguana
```
