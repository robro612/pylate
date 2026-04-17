"""Test embedding anisotropy across models on a small dataset.

Encodes documents with each model, then measures:
- Mean vector norm (1.0 = maximally anisotropic)
- Average pairwise cosine similarity
- Singular value concentration

Usage:
    python scripts/anisotropy_analysis.py
"""

import numpy as np
import torch
from pylate import models


DATASET = "beir/nfcorpus/test"
SAMPLE_SIZE = 5000
SVD_SIZE = 1000

MODELS = [
    ("colbert_kd", "output/modernbert_colbert_kd/final"),
    ("xtr_kd_k256", "output/modernbert_xtr_kd_k256/final"),
    ("colbert_contrastive", "output/modernbert_colbert_contrastive/final"),
    ("xtr_contrastive_k256", "output/modernbert_xtr_contrastive_k256/final"),
    ("gte_moderncolbert", "lightonai/GTE-ModernColBERT-v1"),
    ("xtr_base_en", "robro612/xtr-base-en-pylate"),
]


def load_documents():
    import ir_datasets
    ds = ir_datasets.load(DATASET)
    docs = [doc.text for doc in ds.docs_iter()]
    return docs


def encode_docs(model_name, docs, batch_size=512):
    model = models.ColBERT(model_name_or_path=model_name)
    embeddings = model.encode(
        sentences=docs,
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=True,
    )
    # Flatten to (n_tokens, dim)
    all_tokens = []
    for emb in embeddings:
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        all_tokens.append(emb.astype(np.float32))
    flat = np.concatenate(all_tokens, axis=0)

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return flat


def analyse(name, emb):
    print(f"\n  {name}:")
    print(f"    shape: {emb.shape}")

    # Mean vector norm
    mean_vec = emb.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)

    # Average pairwise cosine sim (sample)
    rng = np.random.default_rng(42)
    n = min(SAMPLE_SIZE, len(emb))
    idx = rng.choice(len(emb), size=n, replace=False)
    sample = emb[idx]
    sims = sample @ sample.T
    np.fill_diagonal(sims, 0)
    avg_sim = sims.sum() / (n * (n - 1))

    # SVD concentration
    svd_n = min(SVD_SIZE, len(emb))
    svd_sample = emb[rng.choice(len(emb), size=svd_n, replace=False)]
    _, svs, _ = np.linalg.svd(svd_sample, full_matrices=False)
    sv_sq = svs ** 2
    total = sv_sq.sum()
    top1_pct = sv_sq[0] / total * 100
    top10_pct = sv_sq[:10].sum() / total * 100
    eff_dim = total ** 2 / (sv_sq ** 2).sum()

    print(f"    mean vector norm:       {mean_norm:.4f}")
    print(f"    avg pairwise cosine:    {avg_sim:.4f}")
    print(f"    SV top-1:               {top1_pct:.1f}%")
    print(f"    SV top-10:              {top10_pct:.1f}%")
    print(f"    effective dimensionality: {eff_dim:.1f} / {emb.shape[1]}")

    return {
        "mean_norm": mean_norm,
        "avg_cos_sim": avg_sim,
        "sv_top1_pct": top1_pct,
        "sv_top10_pct": top10_pct,
        "eff_dim": eff_dim,
    }


def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DATASET}")

    all_stats = {}
    for short_name, model_path in MODELS:
        print(f"\nEncoding with {short_name} ({model_path})...")
        emb = encode_docs(model_path, docs)
        all_stats[short_name] = analyse(short_name, emb)

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'model':<25s} {'mean_norm':>10s} {'avg_cos':>10s} {'SV top1%':>10s} {'SV top10%':>10s} {'eff_dim':>10s}")
    print("-" * 80)
    for name, s in all_stats.items():
        print(f"{name:<25s} {s['mean_norm']:>10.4f} {s['avg_cos_sim']:>10.4f} "
              f"{s['sv_top1_pct']:>9.1f}% {s['sv_top10_pct']:>9.1f}% {s['eff_dim']:>10.1f}")


if __name__ == "__main__":
    main()
