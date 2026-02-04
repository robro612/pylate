"""Quick test for imputation strategies."""
from pylate.rank import score_xtr

query_doc_ids = [
    ["doc1", "doc2", "doc3", "doc4", "doc5"],  # Retrieved for query token 0
    ["doc2", "doc3", "doc4", "doc6", "doc7"],  # Retrieved for query token 1
]
query_scores = [
    [0.9, 0.7, 0.5, 0.3, 0.1],  # Scores for query token 0
    [0.8, 0.6, 0.4, 0.2, 0.05],  # Scores for query token 1
]

# Test default (should be "min")
print("default (no imputation arg):")
results_default = score_xtr(query_doc_ids, query_scores, k=5)
for r in results_default:
    print(f"  {r['id']}: {r['score']:.4f}")

for imp in ["min", "zero", "mean", "percentile", "power_law"]:
    results = score_xtr(query_doc_ids, query_scores, k=5, imputation=imp)
    print(f"\n{imp}:")
    for r in results:
        print(f"  {r['id']}: {r['score']:.4f}")

# Verify default matches min
results_min = score_xtr(query_doc_ids, query_scores, k=5, imputation="min")
assert results_default == results_min, "Default should match 'min' imputation"
print("\n✓ Default behavior matches 'min' imputation")


# Test that power-law imputation produces values lower than min
print("\n--- Power-law imputation test ---")
from pylate.rank.rank import _compute_imputation_scores
import torch

# Test with the same query_scores
power_law_imputations = _compute_imputation_scores(
    query_scores=query_scores,
    imputation="power_law",
    percentile=10.0,
    power_law_multiplier=100.0,
    device="cpu",
)

min_imputations = _compute_imputation_scores(
    query_scores=query_scores,
    imputation="min",
    percentile=10.0,
    power_law_multiplier=100.0,
    device="cpu",
)

print("Per-token imputation values:")
for i, (pl_val, min_val) in enumerate(zip(power_law_imputations, min_imputations)):
    print(f"  Token {i}: power_law={pl_val.item():.6f}, min={min_val.item():.6f}")

# Assert power-law values are strictly lower than min for well-behaved distributions
for i, (pl_val, min_val) in enumerate(zip(power_law_imputations, min_imputations)):
    assert pl_val < min_val, (
        f"Token {i}: power_law ({pl_val.item():.6f}) should be < min ({min_val.item():.6f})"
    )
print("✓ Power-law imputation values are lower than min for all tokens")


# Test with more varied score distributions
print("\n--- Edge case tests ---")

# Edge case 1: Very steep drop-off (should extrapolate to very small value)
steep_scores = [[0.95, 0.5, 0.1, 0.01, 0.001]]
steep_pl = _compute_imputation_scores(steep_scores, "power_law", 10.0, 100.0, "cpu")
steep_min = _compute_imputation_scores(steep_scores, "min", 10.0, 100.0, "cpu")
print(f"Steep drop-off: power_law={steep_pl[0].item():.8f}, min={steep_min[0].item():.8f}")
assert steep_pl[0] < steep_min[0], "Steep: power_law should be < min"
print("✓ Steep drop-off case passes")

# Edge case 2: Gradual drop-off (power-law should still be lower)
gradual_scores = [[0.9, 0.85, 0.8, 0.75, 0.7]]
gradual_pl = _compute_imputation_scores(gradual_scores, "power_law", 10.0, 100.0, "cpu")
gradual_min = _compute_imputation_scores(gradual_scores, "min", 10.0, 100.0, "cpu")
print(f"Gradual drop-off: power_law={gradual_pl[0].item():.6f}, min={gradual_min[0].item():.6f}")
assert gradual_pl[0] < gradual_min[0], "Gradual: power_law should be < min"
print("✓ Gradual drop-off case passes")

# Edge case 3: Only 2 scores (minimum for power-law fit)
two_scores = [[0.8, 0.2]]
two_pl = _compute_imputation_scores(two_scores, "power_law", 10.0, 100.0, "cpu")
two_min = _compute_imputation_scores(two_scores, "min", 10.0, 100.0, "cpu")
print(f"Two scores only: power_law={two_pl[0].item():.6f}, min={two_min[0].item():.6f}")
assert two_pl[0] <= two_min[0], "Two scores: power_law should be <= min"
print("✓ Two scores case passes")

# Edge case 4: Single score (should fall back to min)
single_scores = [[0.5]]
single_pl = _compute_imputation_scores(single_scores, "power_law", 10.0, 100.0, "cpu")
single_min = _compute_imputation_scores(single_scores, "min", 10.0, 100.0, "cpu")
print(f"Single score: power_law={single_pl[0].item():.6f}, min={single_min[0].item():.6f}")
assert single_pl[0] == single_min[0], "Single score: power_law should fall back to min"
print("✓ Single score fallback case passes")

print("\n✓ All power-law imputation tests passed!")
