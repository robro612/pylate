"""
Test that unnormalized embeddings provide better discrimination for importance scoring.
"""

import torch
import torch.nn.functional as F

# Create sample embeddings with different magnitudes
# Token 1: high magnitude (important)
# Token 2: medium magnitude
# Token 3: low magnitude (unimportant)
unnormalized_emb = torch.tensor([
    [3.0, 4.0],  # norm = 5.0 (important)
    [1.5, 2.0],  # norm = 2.5 (medium)
    [0.3, 0.4],  # norm = 0.5 (unimportant)
])

normalized_emb = F.normalize(unnormalized_emb, p=2, dim=-1)

print("=" * 80)
print("Unnormalized Embeddings:")
print("=" * 80)
print(f"Token 1: {unnormalized_emb[0].tolist()} -> L2 norm = {torch.norm(unnormalized_emb[0]).item():.3f}")
print(f"Token 2: {unnormalized_emb[1].tolist()} -> L2 norm = {torch.norm(unnormalized_emb[1]).item():.3f}")
print(f"Token 3: {unnormalized_emb[2].tolist()} -> L2 norm = {torch.norm(unnormalized_emb[2]).item():.3f}")

print("\n" + "=" * 80)
print("Normalized Embeddings:")
print("=" * 80)
print(f"Token 1: {normalized_emb[0].tolist()} -> L2 norm = {torch.norm(normalized_emb[0]).item():.3f}")
print(f"Token 2: {normalized_emb[1].tolist()} -> L2 norm = {torch.norm(normalized_emb[1]).item():.3f}")
print(f"Token 3: {normalized_emb[2].tolist()} -> L2 norm = {torch.norm(normalized_emb[2]).item():.3f}")

print("\n" + "=" * 80)
print("Importance Scores (L2 norm only):")
print("=" * 80)

# Compute importance scores using L2 norm
unnorm_scores = torch.norm(unnormalized_emb, dim=-1)
norm_scores = torch.norm(normalized_emb, dim=-1)

print(f"Unnormalized: {unnorm_scores.tolist()}")
print(f"Normalized:   {norm_scores.tolist()}")

print("\n" + "=" * 80)
print("Discrimination Power:")
print("=" * 80)
print(f"Unnormalized score range: {unnorm_scores.min().item():.3f} - {unnorm_scores.max().item():.3f}")
print(f"Normalized score range:   {norm_scores.min().item():.3f} - {norm_scores.max().item():.3f}")
print(f"\nUnnormalized provides {(unnorm_scores.max() / unnorm_scores.min()).item():.1f}x discrimination")
print(f"Normalized provides {(norm_scores.max() / norm_scores.min()).item():.1f}x discrimination")

print("\n" + "=" * 80)
print("Conclusion:")
print("=" * 80)
print("✓ Unnormalized embeddings: L2 norm varies significantly (0.5 to 5.0)")
print("✓ Normalized embeddings: L2 norm is always ~1.0 (no discrimination)")
print("\n✅ Using unnormalized embeddings for importance scoring is CRITICAL!")

