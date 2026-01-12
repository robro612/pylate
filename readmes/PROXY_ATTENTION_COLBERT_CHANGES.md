# ProxyAttentionColBERT - Documentation Changes

**Date**: 2026-01-12

## Summary

Added comprehensive documentation for the `ProxyAttentionColBERT` model and its training script.

---

## Changes Made

### 1. New Documentation File Created

**File**: `pylate/examples/train/PROXY_ATTENTION_COLBERT.md`

A detailed markdown documentation explaining the ProxyAttentionColBERT architecture, including:

#### Section 1: Overview
- High-level explanation of what ProxyAttentionColBERT does
- Key idea: using learnable proxy tokens to compress document representations
- Benefits: fixed-size output (~10x storage reduction)

#### Section 2: Mathematical Formulation
- Formal definitions with LaTeX equations covering:
  - Proxy token embeddings: $\Psi \in \mathbb{R}^{n_\psi \times d}$
  - Document encoding with proxy token concatenation
  - Saliency computation from attention weights
  - Hard top-k selection algorithm
  - Cluster pooling with weighted averaging
  - Final output normalization

#### Section 3: Code Walkthrough
- Detailed mapping of components to source code locations:
  - `ProxyEmbeddingsModule` (lines 22-129 in ProxyAttentionColBERT.py)
  - `_compute_saliency_from_attention` (lines 405-466)
  - `_hard_select` and `_cluster_pool` methods
  - `ProxyAttentionDistillation` loss function
  - Training script structure

#### Section 4: Training Configuration
- Model configuration table with all parameters
- Training hyperparameters (batch size, learning rate, epochs)
- Loss function details (KL divergence with teacher scores)
- Expected model output characteristics

#### Section 5: Architecture Diagram
- ASCII visualization of the complete pipeline:
  - Query encoding path (standard ColBERT)
  - Document encoding path (with proxy attention)
  - Scoring mechanism (MaxSim)

---

## Files Referenced

The documentation covers these existing source files:

| File | Description |
|------|-------------|
| `pylate/pylate/models/ProxyAttentionColBERT.py` | Main model implementation (839 lines) |
| `pylate/pylate/losses/proxy_attention_distillation.py` | Distillation loss for training (143 lines) |
| `pylate/examples/train/proxy_attention_colbert.py` | Training script (99 lines) |

---

## Key Components Documented

### ProxyAttentionColBERT Model

| Component | Purpose |
|-----------|---------|
| `ProxyEmbeddingsModule` | Stores learnable proxy tokens as `nn.Parameter` |
| `_enable_eager_for_last_layer()` | Enables attention weight output for the final transformer layer |
| `_append_proxy_tokens()` | Concatenates proxy embeddings to document embeddings |
| `_compute_saliency_from_attention()` | Computes saliency scores from proxy→document attention |
| `_hard_select()` | Selects top-k salient tokens as centroids |
| `_cluster_pool()` | Aggregates tokens around selected centroids |
| `forward()` | Routes queries to ColBERT, documents to proxy attention |
| `save()` / `load()` | Serialization with proxy embeddings |

### ProxyAttentionDistillation Loss

| Component | Purpose |
|-----------|---------|
| `score_metric` | ColBERT MaxSim scoring function |
| `loss_function` | KL Divergence loss |
| `forward()` | Computes distillation loss between model and teacher scores |

---

## Configuration Defaults

From the training script (`proxy_attention_colbert.py`):

```python
# Model
model_name = "lightonai/GTE-ModernColBERT-v1"
document_length = 300
num_proxy_tokens = 32
num_select_tokens = 32
use_cluster_pooling = True
proxy_tau = 1.0

# Training
batch_size = 16
learning_rate = 3e-5
num_train_epochs = 3
bf16 = True
eval_steps = 500
save_steps = 5000
```

---

## Usage

To view the documentation:
```bash
cat pylate/examples/train/PROXY_ATTENTION_COLBERT.md
```

To run the training script:
```bash
python pylate/examples/train/proxy_attention_colbert.py
```

---

## Related Files

- Main README: `pylate/readmes/README.md`
- Model tests: `pylate/tests/proxy_attention/`
- Compression experiments: `pylate/experiments/compression/`

