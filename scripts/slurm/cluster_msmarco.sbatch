#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=460G
#SBATCH -J ms_cluster
# Cluster the encoded MSMARCO corpus at 1% of tokens. Method via $1 (tac|pgc).
# PGC params match the lotte winner (mult=40, topm=4, tau=0.05, n_iter=5); TAC n_iter=10.
# --mem=460G: msmarco (~3.7x lotte docs) TAC peaks higher than lotte's ~170 GiB; a 500G
# node gives headroom (TAC materializes full per-head-token buffers). Submit with
# --dependency=afterok:<encode_jobid> so it starts only after a successful encode.
set -euo pipefail   # fail loudly: a killed/erroring cluster_only.py must NOT be masked as
                    # COMPLETED (the original bug: set -u only -> fell through to echo, exit 0,
                    # so the dependent build's afterok "succeeded" then failed on empty output).
PROJ=/exp/rjha/pylate-pgc
cd "$PROJ"
METHOD="${1:?usage: sbatch cluster_msmarco.sbatch <tac|pgc>}"

D=$(ls -d "$PROJ"/embeddings_cache/beir_msmarco_dev/*/nopool_fp16/docs 2>/dev/null | head -1)
if [ -z "$D" ] || [ ! -d "$D" ]; then echo "ERROR: msmarco docs shards not found under embeddings_cache/beir_msmarco_dev"; exit 1; fi

# 1% of total corpus tokens -> centroid budget (matches the lotte iso-1% setup)
NTOK=$(uv run --no-sync python -c "
import glob,os,numpy as np
d='$D'; t=0
for f in sorted(glob.glob(os.path.join(d,'doc_shard_*.npy'))):
    if '.doclens.' in f or '.token_ids.' in f: continue
    t += np.load(f, mmap_mode='r').shape[0]
print(t)
")
TC=$(( NTOK / 100 ))
OUT="$PROJ/clusterings/msmarco_${METHOD}"
echo "=== MSMARCO $METHOD clustering: tokens=$NTOK -> 1% centroids=$TC, shards=$D ==="

if [ "$METHOD" = pgc ]; then
  uv run --no-sync python scripts/cluster_only.py --shard-dir "$D" --method pgc \
    --total-centroids "$TC" --n-iter 5 \
    --pgc-sample-multiplier 40 --pgc-assign-topm 4 --pgc-assign-temp 0.05 \
    --out-dir "$OUT"
else
  uv run --no-sync python scripts/cluster_only.py --shard-dir "$D" --method tac \
    --total-centroids "$TC" --n-iter 10 --out-dir "$OUT"
fi
echo "=== MSMARCO $METHOD clustering done $(date) -> $OUT ==="
