#!/usr/bin/env bash
# QM9 latent OOD certificate on a DeltaAI GH200.
# Trains a baseline, fits droplet_core on the molecular embeddings, then shows (1) selective-prediction
# calibration on real val and (2) OOD detection vs position-corrupted geometries.
# Prereq: bash deltaai/setup_env.sh (once). Submit:  sbatch deltaai/qm9_ood.sbatch
# Output -> ./qm9-ood_<jobid>.out + results/ood_certificate.png + results/ood_certificate.json
#SBATCH --account=bhry-dtai-gh
#SBATCH --job-name=qm9-ood
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.out

set -euo pipefail
WORK=/work/nvme/bhry/crodriguez3

module load python/miniforge3_pytorch/2.10.0
source "$WORK/qm9-venv/bin/activate"
cd "$WORK/qm9/qm9-sota"
export QM9_ROOT="$WORK/qm9-data/QM9"
export OOD_MODEL="${OOD_MODEL:-painn}"
export OOD_EPOCHS="${OOD_EPOCHS:-20}"
export OOD_BUDGET="${OOD_BUDGET:-student_t}"

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/ood_certificate.py
echo "done: $(date)  -> results/ood_certificate.png"
