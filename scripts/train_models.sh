#!/bin/bash
#SBATCH --job-name=ictir_train
#SBATCH --array=0-4%4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/ictir_train_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=rjha5@jh.edu

# Submit from project root:
#   sbatch scripts/train_models.sh
#
# To skip contrastive and resume at distillation for a specific task:
#   sbatch --export=ALL,FORCE_CONTRASTIVE=0 --array=2 scripts/train_models.sh
# (task indices: 0=colbert, 1=xtr_k128, 2=xtr_k256, 3=xtr_k512, 4=xtr_multi)

set -e
mkdir -p logs

VARIANTS=(colbert xtr_k512 xtr_k256 xtr_k128 xtr_multi)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

case $VARIANT in
  colbert)
    C_LOSS=contrastive_colbert
    D_LOSS=kd_colbert
    C_RUN=modernbert_colbert_contrastive
    D_RUN=modernbert_colbert_kd
    K_TRAIN=""
    ;;
  xtr_k128)
    C_LOSS=contrastive_xtr
    D_LOSS=kd_xtr
    C_RUN=modernbert_xtr_contrastive_k128
    D_RUN=modernbert_xtr_kd_k128
    K_TRAIN="[128]"
    ;;
  xtr_k256)
    C_LOSS=contrastive_xtr
    D_LOSS=kd_xtr
    C_RUN=modernbert_xtr_contrastive_k256
    D_RUN=modernbert_xtr_kd_k256
    K_TRAIN="[256]"
    ;;
  xtr_k512)
    C_LOSS=contrastive_xtr
    D_LOSS=kd_xtr
    C_RUN=modernbert_xtr_contrastive_k512
    D_RUN=modernbert_xtr_kd_k512
    K_TRAIN="[512]"
    ;;
  xtr_multi)
    C_LOSS=contrastive_xtr
    D_LOSS=kd_xtr
    C_RUN=modernbert_xtr_contrastive_multik128-256-512
    D_RUN=modernbert_xtr_kd_multik128-256-512
    K_TRAIN="[128,256,512]"
    ;;
esac

C_ARGS=(loss=$C_LOSS run_name=$C_RUN)
D_ARGS=(--config-name distillation loss=$D_LOSS "model_name=output/${C_RUN}/final" run_name=$D_RUN)
if [ -n "$K_TRAIN" ]; then
    C_ARGS+=("loss.k_train=$K_TRAIN")
    D_ARGS+=("loss.k_train=$K_TRAIN")
fi

# Contrastive phase — skipped automatically if checkpoint exists.
# To force re-run: sbatch --export=ALL,FORCE_CONTRASTIVE=1 --array=N scripts/train_ictir.sh
if [ ! -d "output/${C_RUN}/final" ] || [ "${FORCE_CONTRASTIVE:-0}" = "1" ]; then
    echo "=== [task $SLURM_ARRAY_TASK_ID] contrastive: $C_RUN ==="
    python examples/train/hydra_train.py "${C_ARGS[@]}"
else
    echo "=== [task $SLURM_ARRAY_TASK_ID] contrastive checkpoint exists, skipping: $C_RUN ==="
fi

# Distillation phase
echo "=== [task $SLURM_ARRAY_TASK_ID] distillation: $D_RUN ==="
python examples/train/hydra_train.py "${D_ARGS[@]}"
