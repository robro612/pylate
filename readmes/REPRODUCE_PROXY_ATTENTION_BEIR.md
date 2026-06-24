# Reproducing & Extending Proxy-Attention (Attention-Guided Clustering) on BEIR

This guide documents how to reproduce the **proxy-attention** results from scratch and how
to extend them to new vector budgets (e.g. **8 or 16 vectors per document**), which is what
most follow-up requests ask for. The method is implemented as the **`ProxyAttentionColBERT`**
model: learnable proxy query tokens compute attention saliency over document tokens, the
top-`m` most salient tokens are selected, and (optionally) nearby tokens are pooled into those
centroids — i.e. *attention-guided clustering*.

> **TL;DR** — Everything needed (model, loss, training script, BEIR eval harness, configs,
> tests) is committed. The trained checkpoints are *not* shipped, so each vector budget
> requires training one model. A different budget (8 vs 16) = a separately trained model.
> Read the **Gotchas** section before evaluating — the eval entrypoint needs `.load()`, not
> the bare constructor, or it silently uses random proxy weights.

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

**Training**
- `examples/train/proxy_attention_colbert.py` — CLI training entrypoint (KD on MS MARCO).
- `examples/train/PROXY_ATTENTION_COLBERT.md` — full math + code walkthrough of the method.
- `conf/model/proxy_attention.yaml` — config template for the model variant.

**Evaluation**
- `examples/evaluation/beir_dataset.py` — **full BEIR test eval** with a PLAID index
  (`--model_type proxy_attention`). This is the path that produces per-dataset BEIR numbers.
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
# Optional extras used by some training modes:
#   peft        -> required only for --training_mode lora
```

The base model and KD dataset are pulled from the HuggingFace Hub automatically:
- Base encoder: `Alibaba-NLP/gte-modernbert-base` (paper default) — or `lightonai/GTE-ModernColBERT-v1`
  to start from an already-ColBERT-tuned checkpoint.
- KD training data: `lightonai/ms-marco-en-bge-gemma` (MS MARCO with BGE-Gemma teacher scores).

GPU notes for this cluster: train on L40S/A100/H100 (`srunl40s` / a100 / h100). V100 cannot run
the cu13 torch build. Encoding/eval can run on GPU; PLAID indexing defaults to CPU.

---

## 4. Train the models (one per vector budget)

The output vector count **m** is the `--num_select_tokens` flag and is fixed at *training* time.
To get an 8-vector and a 16-vector model, train two models:

```bash
# 8 vectors/doc
torchrun --nproc_per_node=4 examples/train/proxy_attention_colbert.py \
    --model_name Alibaba-NLP/gte-modernbert-base \
    --num_select_tokens 8  --num_proxy_tokens 8 \
    --use_cluster_pooling \
    --batch_size 24 --n_ways 16 --lr 3e-5 --epochs 3 --bf16 \
    --output_dir output/proxy_attention-S8

# 16 vectors/doc
torchrun --nproc_per_node=4 examples/train/proxy_attention_colbert.py \
    --model_name Alibaba-NLP/gte-modernbert-base \
    --num_select_tokens 16 --num_proxy_tokens 16 \
    --use_cluster_pooling \
    --batch_size 24 --n_ways 16 --lr 3e-5 --epochs 3 --bf16 \
    --output_dir output/proxy_attention-S16
```

Notes:
- Single GPU: drop `torchrun --nproc_per_node=N` and just run `python ...`.
- The final model is written to `<output_dir>/final` (via the model's custom `save()`, which
  writes `proxy_embeddings/` and records the proxy params in `config_sentence_transformers.json`).
- `--num_proxy_tokens` need not equal `--num_select_tokens`, but matching them is the convention
  used here. `num_proxy_tokens` = how many learnable probes compute saliency; `num_select_tokens`
  = how many output vectors are kept.
- Training modes (`--training_mode`): `full` (default), `proxy_only` (freeze everything but the
  proxy tokens — cheapest), `freeze_word_embeddings`, `lora` (needs `peft`). The paper used full
  fine-tuning; `proxy_only` is a fast sanity check.
- Mid-training NanoBEIR scores are logged but are *not* the reported BEIR test numbers.

---

## 5. Evaluate on BEIR

### ⚠️ Required fix before evaluating (read this)

`examples/evaluation/beir_dataset.py` instantiates models with the **plain constructor**:

```python
model = model_class(model_name_or_path=model_name, document_length=300, query_length=...)
```

For `ProxyAttentionColBERT` the constructor **does not** read the saved proxy config and **does
not** load the trained proxy-token weights. It will silently fall back to the defaults
(`num_select_tokens=32`) with **randomly initialized** proxy embeddings — i.e. it will *not*
evaluate your trained 8/16-vector model. Only the `ProxyAttentionColBERT.load(path)` classmethod
restores `num_select_tokens` and the learned proxy embeddings from disk.

Apply this one-line change in `beir_dataset.py` where the model is built (around line 173):

```python
# before
model = model_class(
    model_name_or_path=model_name,
    document_length=300,
    query_length=query_len.get(dataset_name),
)

# after
if model_type == "proxy_attention":
    model = models.ProxyAttentionColBERT.load(
        model_name,
        document_length=300,
        query_length=query_len.get(dataset_name),
    )
else:
    model = model_class(
        model_name_or_path=model_name,
        document_length=300,
        query_length=query_len.get(dataset_name),
    )
```

(You can verify the fix took effect: the encoded docs should average ≈ `num_select_tokens`
vectors/doc in the `print_token_stats` output — 8 or 16, not ~300.)

### Run it

```bash
python examples/evaluation/beir_dataset.py \
    --model_type proxy_attention \
    --model_name_or_path output/proxy_attention-S8/final \
    --index_type plaid \
    --output_dir evaluation_results/proxy_attention-S8 \
    --save_runfile \
    --dataset_name nfcorpus scifact fiqa trec-covid scidocs arguana \
                   webis-touche2020 quora nq hotpotqa fever \
                   climate-fever dbpedia-entity
```

This loads each BEIR dataset, encodes docs (→ 8 vectors/doc) and queries, builds a PLAID index,
retrieves, and writes `evaluation_results/.../overall_results.jsonl` plus per-dataset
`evaluation_results.json` (and `run.json` runfiles with `--save_runfile`). Reported metrics:
`map, ndcg@10, ndcg@100, recall@10, recall@100`. Repeat with `output/proxy_attention-S16/final`.

Per-dataset query lengths are already encoded in the `query_len` dict at the top of the script.

---

## 6. Extending the method

- **Other vector budgets** — train with any `--num_select_tokens` (4, 8, 16, 24, 32, …). One
  model per budget.
- **Pooling variants** — `--no_cluster_pooling` (pure top-k selection, no clustering),
  `cluster_centroid_weight`, `use_attn_weight_cluster_pooling` (saliency-weighted pooling). These
  are constructor args; expose them as CLI flags in `proxy_attention_colbert.py` if you want to
  sweep them (only the four already-wired flags are CLI-exposed today).
- **Saliency temperature** — `--proxy_tau` sharpens/softens the proxy→token attention softmax.
- **Different base encoder** — `--model_name <any ColBERT-compatible encoder>`. Architecture
  support for the attention capture lives in `_get_last_layer` (ModernBERT / BERT / RoBERTa
  styles handled; add a branch for others).
- **New datasets** — add the dataset to `--dataset_name`; set its query length in the `query_len`
  dict in `beir_dataset.py`. `cqadupstack/*` subsets are handled specially (auto-downloaded).

---

## 7. Gotchas & caveats checklist

- [ ] **Use `.load()`, not the constructor**, when evaluating a trained proxy model (Section 5).
      Otherwise you measure a random model at the wrong vector count.
- [ ] **Checkpoints are not in git.** Train first; nothing is reproducible from committed weights.
- [ ] **8 ≠ 16 is two training runs**, not an eval-time flag. The budget is baked in at train time.
- [ ] **NanoBEIR ≠ BEIR.** The in-training evaluator uses small subsets; report numbers from
      `beir_dataset.py`.
- [ ] **Working-tree drift.** As of this writing, several eval-side files have *uncommitted local
      modifications* (`experiments/compression/compression_eval*.py`,
      `experiments/compression/conf/compression_eval.yaml`, `pylate/evaluation/beir.py`, and
      `pylate/models/compression/*.py`). The `ProxyAttentionColBERT` training/eval path
      (`examples/...`) is unaffected, but if you hand this off, commit or stash those diffs so the
      recipient isn't chasing behavior that only exists on a local tree.
- [ ] **GPU/CUDA:** V100 is incompatible with the cu13 torch build; train/encode on L40S/A100/H100.

---

## 8. One-screen recipe

```bash
# 0. install
uv pip install -e .

# 1. train (two budgets)
torchrun --nproc_per_node=4 examples/train/proxy_attention_colbert.py \
    --num_select_tokens 8  --num_proxy_tokens 8  --use_cluster_pooling \
    --output_dir output/proxy_attention-S8
torchrun --nproc_per_node=4 examples/train/proxy_attention_colbert.py \
    --num_select_tokens 16 --num_proxy_tokens 16 --use_cluster_pooling \
    --output_dir output/proxy_attention-S16

# 2. apply the .load() fix in examples/evaluation/beir_dataset.py (Section 5)

# 3. eval on BEIR
python examples/evaluation/beir_dataset.py --model_type proxy_attention \
    --model_name_or_path output/proxy_attention-S8/final \
    --index_type plaid --save_runfile --output_dir evaluation_results/S8 \
    --dataset_name nfcorpus scifact fiqa trec-covid scidocs arguana
```
