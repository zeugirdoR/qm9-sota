#!/usr/bin/env bash
# Phase 2a: label-noise robustness sweep on a DeltaAI GH200.
# For each corrupted-target fraction, train baseline vs fully-engaged droplet on the SAME corrupted data
# (QM9_NOISE_SEED fixed) and compare CLEAN-val MAE. Thesis: the droplet amputates the corrupted molecules
# (bounded influence) so it degrades far less than the baseline as noise rises.
# Prereq: bash deltaai/setup_env.sh (once). Submit:  sbatch deltaai/qm9_noise.sbatch
# Output -> ./qm9-noise_<jobid>.out + results/noise*/ + results/noise_robustness.png
#SBATCH --account=bhry-dtai-gh
#SBATCH --job-name=qm9-noise
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out

set -euo pipefail
WORK=/work/nvme/bhry/crodriguez3

module load python/miniforge3_pytorch/2.10.0
source "$WORK/qm9-venv/bin/activate"
cd "$WORK/qm9/qm9-sota"
export QM9_ROOT="$WORK/qm9-data/QM9"
export QM9_NOISE_SEED=0          # same corrupted molecules for baseline vs droplet at each frac (fair)

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for FRAC in 0.0 0.1 0.2 0.3; do
  echo "================ noise fraction = $FRAC ================"
  QM9_NOISE_FRAC=$FRAC python scripts/train.py \
    --config configs/train/smoke_noise.yaml --loss configs/loss/baseline.yaml \
    --run-name "noise${FRAC}_baseline"
  QM9_NOISE_FRAC=$FRAC python scripts/train.py \
    --config configs/train/smoke_noise.yaml --loss configs/loss/droplet_robust.yaml \
    --run-name "noise${FRAC}_droplet"
  QM9_NOISE_FRAC=$FRAC python scripts/train.py \
    --config configs/train/smoke_noise.yaml --loss configs/loss/droplet_core.yaml \
    --run-name "noise${FRAC}_core"
done

echo "================ collating ================"
python deltaai/plot_noise.py
echo "done: $(date)  -> results/noise_robustness.png"
