# Experiments

This directory contains analysis scripts for reproducing experiments from the XTR paper.

## Experiment 1: Reproduce XTR's Figure 6: Retrieval Performance vs K_token for ColBERT and XTR models using ColBERT and XTR scoring.

**Goal:** Reproduce Figure 6 from the XTR paper comparing retrieval performance.

- **Config:** `conf/eval/experiment_1_*.yaml` (colbert, xtr, colbert_xtr, colbert_colbert)
- **Script:** `eval_model_irds_v2.py` (main evaluation script)
- **Run:** `bash scripts/experiment_1_fig_6.sh`

## Experiment 2: Score Imputation Methods

**Goal:** Compare different imputation strategies for XTR scoring (min, mean, percentile, power_law, zero).

- **Config:** Uses `conf/eval/retrieve/imputation/*.yaml` (no dedicated top-level config)
- **Script:** `eval_model_irds_v2.py` with imputation overrides
- **Run:** `bash scripts/experiment_2_imputation_methods.sh`

## Experiment 3: Token Score Distribution

**Goal:** Analyze the distribution of token-level similarity scores for relevant vs non-relevant documents.

- **Config:** `conf/eval/experiment_3_token_scores.yaml`
- **Script:** `analyze_token_scores.py` (at root)
- **Run:** `bash scripts/experiment_3_analyze_token_scores.sh`

## Experiment 4: Token Rank Analysis (P(Gold | rank k))

**Goal:** Plot the probability that a token retrieved at rank k comes from a relevant document.

- **Config:** `conf/eval/experiment_4_rank_analysis.yaml`
- **Script:** `analyze_token_rank.py` (at root)
- **Run:** `bash scripts/experiment_4_rank_analysis.sh`

## Running Experiments

All scripts should be run from the repository root:

```bash
# Example: Run experiment 4
python analyze_token_rank.py

# With overrides
python analyze_token_rank.py dataset.names=[beir/scifact/test]
```

Or use the bash wrapper scripts:

```bash
bash scripts/experiment_4_rank_analysis.sh
```

## Test Scripts

The `experiments/test/` directory contains unit and integration tests for specific functionality.
