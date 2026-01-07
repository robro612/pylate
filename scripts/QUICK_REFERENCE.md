# Quick Reference Guide

## Common Use Cases

### 1. Run Single Experiment Locally

```bash
# Basic run
./pylate/scripts/run_compression_experiments.sh -d nfcorpus

# With retrieval results saved
./pylate/scripts/run_compression_experiments.sh -d nfcorpus --save-retrieval-results

# With custom output directory
./pylate/scripts/run_compression_experiments.sh -d nfcorpus -o results/my_test
```

### 2. Run Multiple Datasets Locally

```bash
# Default datasets (nfcorpus, scifact, arguana)
./pylate/scripts/run_all_datasets.sh

# Custom datasets
./pylate/scripts/run_all_datasets.sh -d "nfcorpus scifact"

# With retrieval results
./pylate/scripts/run_all_datasets.sh --save-retrieval-results

# Parallel execution (faster)
./pylate/scripts/run_all_datasets.sh --parallel
```

### 3. Submit SLURM Jobs

```bash
# Single job (using sbatch directly)
sbatch --export=DATASET=nfcorpus pylate/scripts/submit_single_job.sh

# Single job with options
sbatch --export=DATASET=scifact,SAVE_RETRIEVAL_RESULTS=true \
       --gpus=2 --mem=128G --time=48:00:00 \
       pylate/scripts/submit_single_job.sh

# Multiple jobs (recommended - uses submit_multiple_jobs.sh)
./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana" --save-retrieval-results

# Multiple datasets with custom resources
./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana" --gpus 2 --mem 128G
```

### 4. Monitor SLURM Jobs

```bash
# Check job status
squeue -u $USER

# Watch job status (updates every second)
watch -n 1 squeue -u $USER

# View logs in real-time
tail -f pylate/logs/compression-exp_*.out

# View error logs
tail -f pylate/logs/compression-exp_*.err
```

### 5. Custom Configurations

```bash
# Custom metrics
./pylate/scripts/run_compression_experiments.sh \
    -d nfcorpus \
    --metrics "map ndcg@10 precision@10 recall@10"

# Custom model
./pylate/scripts/run_compression_experiments.sh \
    -d nfcorpus \
    -m "sentence-transformers/all-MiniLM-L6-v2"

# Custom batch size
./pylate/scripts/run_compression_experiments.sh \
    -d nfcorpus \
    -b 2000

# Flat index instead of PLAID
./pylate/scripts/run_compression_experiments.sh \
    -d nfcorpus \
    -i flat
```

## Script Comparison

| Feature | `run_compression_experiments.sh` | `run_all_datasets.sh` | `submit_single_job.sh` | `submit_multiple_jobs.sh` |
|---------|----------------------------------|----------------------|------------------------|---------------------------|
| Single dataset | ✅ | ❌ | ✅ | ❌ |
| Multiple datasets | ❌ | ✅ | ❌ | ✅ |
| Local execution | ✅ | ✅ | ❌ | ❌ |
| SLURM cluster | ❌ | ❌ | ✅ | ✅ |
| Parallel execution | ❌ | ✅ (optional) | ❌ | ✅ (automatic) |
| Custom output dir | ✅ | ❌ | ❌ | ❌ |
| Custom metrics | ✅ | ❌ | ❌ | ❌ |
| Custom configs file | ✅ | ❌ | ❌ | ❌ |

## Output Locations

```
pylate/results/compression_experiments/
  lightonai_GTE-ModernColBERT-v1/
    nfcorpus/
      experiment_20231215_143022/
        results.jsonl                    # Main results
        summary_table.txt                # Summary table
        runfiles/                        # Ranx runfiles (if --save-runfiles)
        retrieval_results/               # Raw retrieval results (if --save-retrieval-results)
```

## Logs (SLURM only)

```
pylate/logs/
  compression-exp_12345.out            # Standard output
  compression-exp_12345.err            # Error output
```

## Tips & Tricks

### 1. Save Retrieval Results for Analysis

Always use `--save-retrieval-results` if you want to analyze results later:

```bash
./pylate/scripts/run_compression_experiments.sh -d nfcorpus --save-retrieval-results
```

### 2. Run Multiple Datasets Efficiently

Use parallel execution for faster results (if you have enough resources):

```bash
./pylate/scripts/run_all_datasets.sh --parallel
```

### 3. Submit Multiple SLURM Jobs

```bash
# Submit jobs for all datasets in one command
./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana fiqa trec-covid" --save-retrieval-results

# Check all jobs
squeue -u $USER
```

### 4. Custom Job Names for SLURM

```bash
# Single job (override default job name)
sbatch --export=DATASET=nfcorpus \
       --job-name=nfcorpus-exp \
       pylate/scripts/submit_single_job.sh

# Multiple jobs (prefix will be used)
./pylate/scripts/submit_multiple_jobs.sh \
    -d "nfcorpus scifact" \
    --job-name "my-exp" \
    --save-retrieval-results
# Jobs will be named: my-exp-nfcorpus, my-exp-scifact
```

### 5. Organize Experiments

Use custom output directories to organize different experiment runs:

```bash
./pylate/scripts/run_compression_experiments.sh \
    -d nfcorpus \
    -o results/baseline_experiment

./pylate/scripts/run_compression_experiments.sh \
    -d nfcorpus \
    -o results/attention_pooling_only
```

## Troubleshooting

### Script not executable

```bash
chmod +x pylate/scripts/*.sh
```

### SLURM job fails immediately

Check logs:
```bash
cat pylate/logs/compression-exp_*.err
```

### Out of memory

Increase memory limit:
```bash
./pylate/scripts/submit_slurm_job.sh -d nfcorpus --mem 64G
```

### Job timeout

Increase time limit:
```bash
./pylate/scripts/submit_slurm_job.sh -d nfcorpus --time "48:00:00"
```

### Need more GPUs

```bash
./pylate/scripts/submit_slurm_job.sh -d nfcorpus --gpu "a100:2"
```

