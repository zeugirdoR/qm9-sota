#!/usr/bin/env bash
# Phase 1: train the PaiNN equivariant backbone on full QM9 (DeltaAI GH200).
# Goal: a real, competitive backbone whose embedding unblocks the OOD certificate (vs the toy 0.32 MAE).
# Prereq: bash deltaai/setup_env.sh (once). Submit:  sbatch deltaai/qm9_painn.sbatch
# Output -> ./qm9-painn_<jobid>.out + results/painn_baseline_gh200/ (epoch_log.csv, summary.json, best_model.pt)
#SBATCH --account=bhry-dtai-gh
#SBATCH --job-name=qm9-painn
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
WORK=/work/nvme/bhry/crodriguez3

module load python/miniforge3_pytorch/2.10.0
source "$WORK/qm9-venv/bin/activate"
cd "$WORK/qm9/qm9-sota"
export QM9_ROOT="$WORK/qm9-data/QM9"

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== PaiNN equivariance self-test (must PASS before training) ==="
PYTHONPATH=src python src/qm9sota/models/painn.py

echo "=== train PaiNN on full QM9 (20 epochs) ==="
python scripts/train.py --config configs/train/painn.yaml \
  --loss configs/loss/baseline.yaml --run-name painn_baseline_gh200

echo "done: $(date)  -> results/painn_baseline_gh200/summary.json"
