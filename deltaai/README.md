# Running qm9-sota on NCSA DeltaAI (GH200)

This is the DeltaAI (GH200, aarch64) runner for qm9-sota — the compute-compartmentalized sibling of the
Colab path. The alignment project (`TOI-droplet`) and this one use **separate** WORK dirs, venvs, and Slurm
jobs on the same allocation `bhry-dtai-gh`.

Layout on DeltaAI:
- repo:  `/work/nvme/bhry/crodriguez3/qm9/qm9-sota`
- venv:  `/work/nvme/bhry/crodriguez3/qm9-venv`     (PyG stack; separate from the alignment venv)
- data:  `/work/nvme/bhry/crodriguez3/qm9-data/QM9` (via the `QM9_ROOT` env override in `data/qm9.py`)

## First time
From a laptop with the `deltaai` helper:
```
deltaai --qm9 'git clone https://github.com/zeugirdoR/qm9-sota.git /work/nvme/bhry/crodriguez3/qm9/qm9-sota'   # or pull
deltaai --qm9 'bash deltaai/setup_env.sh'     # Gate-A aarch64 build + stage QM9 (login node, ~few min)
```

## Run
```
deltaai --qm9 'sbatch deltaai/qm9_smoke.sbatch'    # A0 baseline vs A1b gentle droplet on a GH200
deltaai --qm9 squeue --me
deltaai --qm9 'cat qm9-smoke_<jobid>.out'
```

`deltaai --qm9` (no command) drops you into an interactive `qm9` tmux session, in the repo with the
qm9-venv active.

## Notes
- **No RDKit** in this venv on purpose (PyG preprocessed QM9 path; rdkit can break QM9 processing).
- Compute nodes are offline — `setup_env.sh` stages the dataset on the login node first.
- This is Phase 0/1 (build + competitive baseline). Phase 2 swaps in the real invariant droplet core from
  `TOI-droplet` + a latent OOD certificate; Phase 3 is droplet-gated generation.
