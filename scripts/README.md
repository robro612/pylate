# Pylate Experiment Scripts

Bash scripts for running compression experiments easily.

## Available Scripts

### 1. `run_compression_experiments.sh`

Run a single compression experiment with customizable options.

**Usage:**
```bash
./pylate/scripts/run_compression_experiments.sh [OPTIONS]
```

**Options:**
- `-d, --dataset DATASET` - Dataset name (default: nfcorpus)
- `-m, --model MODEL` - Model name (default: lightonai/GTE-ModernColBERT-v1)
- `-i, --index-type TYPE` - Index type: flat or plaid (default: plaid)
- `-o, --output-dir DIR` - Output directory (default: auto-generated)
- `-b, --batch-size SIZE` - Batch size for encoding (default: 1000)
- `--save-runfiles` - Save ranx runfiles
- `--save-retrieval-results` - Save raw retrieval results
- `--metrics METRICS` - Space-separated metrics
- `--configs-file FILE` - Path to custom configs JSONL file
- `-h, --help` - Show help message

**Examples:**
```bash
# Run on nfcorpus with default settings
./pylate/scripts/run_compression_experiments.sh

# Run on scifact with retrieval results saved
./pylate/scripts/run_compression_experiments.sh -d scifact --save-retrieval-results

# Run on arguana with custom output directory
./pylate/scripts/run_compression_experiments.sh -d arguana -o results/my_experiment

# Run with custom metrics
./pylate/scripts/run_compression_experiments.sh --metrics "map ndcg@10 precision@10"
```

---

### 2. `run_all_datasets.sh`

Run compression experiments on multiple datasets sequentially or in parallel.

**Usage:**
```bash
./pylate/scripts/run_all_datasets.sh [OPTIONS]
```

**Options:**
- `-d, --datasets DATASETS` - Space-separated dataset names (default: nfcorpus scifact arguana)
- `-m, --model MODEL` - Model name (default: lightonai/GTE-ModernColBERT-v1)
- `-i, --index-type TYPE` - Index type: flat or plaid (default: plaid)
- `--save-runfiles` - Save ranx runfiles
- `--save-retrieval-results` - Save raw retrieval results
- `--parallel` - Run datasets in parallel (experimental)
- `-h, --help` - Show help message

**Examples:**
```bash
# Run on default datasets (nfcorpus, scifact, arguana)
./pylate/scripts/run_all_datasets.sh

# Run on specific datasets with retrieval results saved
./pylate/scripts/run_all_datasets.sh -d "nfcorpus scifact" --save-retrieval-results

# Run on all BEIR datasets
./pylate/scripts/run_all_datasets.sh -d "nfcorpus scifact arguana fiqa trec-covid"

# Run in parallel (faster but uses more resources)
./pylate/scripts/run_all_datasets.sh --parallel
```

---

### 3. `submit_single_job.sh`

SLURM batch script for submitting a single compression experiment. This script has `#SBATCH` directives at the top and is meant to be submitted with `sbatch` or called by `submit_multiple_jobs.sh`.

**Default SLURM Configuration:**
- Job name: `compression-exp`
- Partition: `h100,a100`
- GPUs: `1`
- CPUs: `8`
- Memory: `80G`
- Time limit: `24:00:00`
- Output: `pylate/logs/compression-exp_%j.out`
- Error: `pylate/logs/compression-exp_%j.err`

**Usage:**
```bash
# Submit with sbatch (pass dataset via --export)
sbatch --export=DATASET=nfcorpus pylate/scripts/submit_single_job.sh

# Submit with custom SLURM options
sbatch --export=DATASET=scifact,SAVE_RETRIEVAL_RESULTS=true \
       --gpus=2 --mem=128G --time=48:00:00 \
       pylate/scripts/submit_single_job.sh

# Or use submit_multiple_jobs.sh (recommended for multiple datasets)
./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana"
```

**Environment Variables (set via `--export`):**
- `DATASET` - Dataset name (required)
- `MODEL` - Model name (default: lightonai/GTE-ModernColBERT-v1)
- `INDEX_TYPE` - Index type: flat or plaid (default: plaid)
- `SAVE_RUNFILES` - Set to "true" to save ranx runfiles (default: false)
- `SAVE_RETRIEVAL_RESULTS` - Set to "true" to save raw retrieval results (default: false)

**Examples:**
```bash
# Basic submission
sbatch --export=DATASET=nfcorpus pylate/scripts/submit_single_job.sh

# With retrieval results
sbatch --export=DATASET=scifact,SAVE_RETRIEVAL_RESULTS=true \
       pylate/scripts/submit_single_job.sh

# With custom GPU and time
sbatch --export=DATASET=arguana,SAVE_RETRIEVAL_RESULTS=true \
       --gpus=2 --mem=128G --time=48:00:00 \
       pylate/scripts/submit_single_job.sh

# With all options
sbatch --export=DATASET=fiqa,MODEL=lightonai/GTE-ModernColBERT-v1,INDEX_TYPE=plaid,SAVE_RETRIEVAL_RESULTS=true \
       --partition=a100 --gpus=2 --cpus-per-task=16 --mem=128G --time=48:00:00 \
       pylate/scripts/submit_single_job.sh
```

---

### 4. `submit_multiple_jobs.sh`

Submit multiple compression experiments as SLURM jobs (one job per dataset). This script loops over datasets and calls `sbatch` with `submit_single_job.sh` for each dataset.

**Usage:**
```bash
./pylate/scripts/submit_multiple_jobs.sh [OPTIONS]
```

**SLURM Options:**
- `--job-name NAME` - Job name prefix (default: compression-exp)
  - Each job will be named: `{prefix}-{dataset}`
- `--partition PARTITION` - Partition name (default: h100,a100)
- `--gpus NUM` - Number of GPUs (default: 1)
- `--cpus NUM` - Number of CPUs (default: 8)
- `--mem MEMORY` - Memory limit (default: 80G)
- `--time TIME` - Time limit (default: 24:00:00)
- `--exclude NODES` - Nodes to exclude (optional, e.g., c007,h02)

**Experiment Options:**
- `-d, --datasets DATASETS` - Space-separated dataset names (required)
  - Example: `"nfcorpus scifact arguana"`
- `-m, --model MODEL` - Model name (default: lightonai/GTE-ModernColBERT-v1)
- `-i, --index-type TYPE` - Index type: flat or plaid (default: plaid)
- `--save-runfiles` - Save ranx runfiles
- `--save-retrieval-results` - Save raw retrieval results
- `-h, --help` - Show help message

**How it works:**
For each dataset, it runs:
```bash
sbatch --job-name=compression-exp-{dataset} \
       --partition=h100,a100 \
       --gpus=1 \
       --cpus-per-task=8 \
       --mem=80G \
       --time=24:00:00 \
       --export=DATASET={dataset},MODEL={model},... \
       pylate/scripts/submit_single_job.sh
```

**Examples:**
```bash
# Submit jobs for multiple datasets
./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana"

# Submit jobs with retrieval results saved
./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana" --save-retrieval-results

# Submit jobs with custom resources
./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact" --gpus 2 --mem 128G --time "48:00:00"

# Submit jobs for all BEIR datasets
./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana fiqa trec-covid" --save-retrieval-results

# Submit jobs with custom partition and exclude nodes
./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact" --partition a100 --exclude "c007,h02" --save-retrieval-results
```

**Monitoring:**
```bash
# Check job status
squeue -u $USER

# View logs (while job is running or after completion)
tail -f pylate/logs/compression-exp_<job_id>.out
tail -f pylate/logs/compression-exp_<job_id>.err
```

---

## Quick Start

### Local Execution

```bash
# Single dataset
./pylate/scripts/run_compression_experiments.sh -d nfcorpus --save-retrieval-results

# Multiple datasets
./pylate/scripts/run_all_datasets.sh -d "nfcorpus scifact arguana" --save-retrieval-results
```

### SLURM Cluster

```bash
# Submit single job
./pylate/scripts/submit_single_job.sh -d nfcorpus --save-retrieval-results

# Submit multiple jobs (one per dataset)
./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana" --save-retrieval-results
```

---

## Output Structure

All scripts generate results in the following structure:

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

---

## Tips

1. **Save retrieval results** for later analysis:
   ```bash
   ./pylate/scripts/run_compression_experiments.sh -d nfcorpus --save-retrieval-results
   ```

2. **Use custom output directory** to organize experiments:
   ```bash
   ./pylate/scripts/run_compression_experiments.sh -d nfcorpus -o results/my_experiment_v1
   ```

3. **Run multiple datasets in parallel** (if you have enough resources):
   ```bash
   ./pylate/scripts/run_all_datasets.sh --parallel
   ```

4. **Monitor SLURM jobs**:
   ```bash
   watch -n 1 squeue -u $USER
   ```

5. **Check logs in real-time**:
   ```bash
   tail -f pylate/logs/compression-exp_*.out
   ```

