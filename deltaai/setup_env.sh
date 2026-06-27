#!/usr/bin/env bash
# QM9 stack — one-time DeltaAI environment build + aarch64 Gate-A + dataset staging.
# Run ONCE on a dtai-login node (needs internet):   bash deltaai/setup_env.sh
#
# Builds a SEPARATE venv from the alignment project (different dep stack: PyTorch Geometric).
# Deliberately NO rdkit — the repo uses PyG's preprocessed QM9 path, and rdkit can break QM9
# processing (see src/qm9sota/data/qm9.py warning). Add rdkit/e3nn later, in this venv, only
# when the equivariant / generative phases need them.
set -euo pipefail

WORK=/work/nvme/bhry/crodriguez3
QM9_REPO="$WORK/qm9/qm9-sota"
VENV="$WORK/qm9-venv"
export QM9_ROOT="$WORK/qm9-data/QM9"

echo "=== [Gate-A] aarch64 build for the QM9/PyG stack ==="
uname -m
module load python/miniforge3_pytorch/2.10.0
python -m venv --system-site-packages "$VENV"      # reuse the module's aarch64 torch
source "$VENV/bin/activate"
pip install --upgrade -q pip
pip install -q -e "$QM9_REPO"                       # pulls numpy/pandas/pyyaml/tqdm + the package
pip install -q torch_geometric                      # the one real aarch64 unknown to retire here

python - <<'PY'
import torch, torch_geometric
print("torch", torch.__version__, "| PyG", torch_geometric.__version__, "| cuda:", torch.cuda.is_available())
from torch_geometric.loader import DataLoader      # exercise PyG imports on aarch64
from torch_geometric.datasets import QM9
print("[Gate-A] PyG imports OK on aarch64")
PY

echo "=== staging QM9 (download+preprocess on the login node; compute nodes are offline) ==="
python - <<PY
import os
from torch_geometric.datasets import QM9
ds = QM9(root=os.environ["QM9_ROOT"])
print("QM9 staged:", len(ds), "molecules ->", ds[0])
PY

echo "=== QM9 env ready: $VENV  | data: $QM9_ROOT ==="
