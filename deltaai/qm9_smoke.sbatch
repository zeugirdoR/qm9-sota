#!/usr/bin/env bash
# QM9 smoke on a DeltaAI GH200: reproduce the A0 baseline vs A1b gentle-droplet comparison.
# Prereq: bash deltaai/setup_env.sh  (once, on a login node).
# Submit from the repo root:   sbatch deltaai/qm9_smoke.sbatch
# Output -> ./qm9-smoke_<jobid>.out + results/<run-name>/  (epoch_log.csv, summary.json, per-target MAE)
#SBATCH --account=bhry-dtai-gh
#SBATCH --job-name=qm9-smoke
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -euo pipefail
WORK=/work/nvme/bhry/crodriguez3

module load python/miniforge3_pytorch/2.10.0
source "$WORK/qm9-venv/bin/activate"
cd "$WORK/qm9/qm9-sota"
export QM9_ROOT="$WORK/qm9-data/QM9"

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== A0 baseline (smooth-L1, 5 epochs) ==="
python scripts/train.py --config configs/train/smoke.yaml \
  --loss configs/loss/baseline.yaml --run-name A0_baseline_gh200

echo "=== A1b gentle droplet (5 epochs) ==="
python scripts/train.py --config configs/train/smoke.yaml \
  --loss configs/loss/droplet_gentle.yaml --run-name A1b_gentle_droplet_gh200

echo "done: $(date)  -> results/A0_baseline_gh200 , results/A1b_gentle_droplet_gh200"
