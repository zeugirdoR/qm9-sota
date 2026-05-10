# QM9-SOTA

Reproducible Colab-first research stack for QM9 molecular property prediction and Droplet-style bounded-information losses.

## Current status

This repository begins with a smoke-test stack, not a SOTA model:

- Dataset: PyTorch Geometric QM9 preprocessed dataset
- Molecules: 130,831
- Targets: 19
- Baseline model: `TinyRadialMPNN`
- Primary smoke metric: mean normalized validation MAE across 19 targets
- Canonical runtime: Google Colab GPU

The immediate goal is reproducibility:

1. Reproduce the tiny baseline from a script.
2. Run the gentle Droplet schedule from a script.
3. Record every result in `experiments/` and `results/`.

## Quick Colab launch

```python
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/zeugirdoR/qm9-sota.git"
ROOT = Path("/content/qm9-sota")

if ROOT.exists():
    subprocess.check_call(["git", "-C", str(ROOT), "pull"])
else:
    subprocess.check_call(["git", "clone", REPO_URL, str(ROOT)])

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "environment" / "requirements-colab.txt")])
```

Run a baseline:

```bash
python /content/qm9-sota/scripts/train.py \
  --config /content/qm9-sota/configs/train/smoke.yaml \
  --loss /content/qm9-sota/configs/loss/baseline.yaml \
  --run-name A0_paired_baseline
```

Run the gentle Droplet schedule:

```bash
python /content/qm9-sota/scripts/train.py \
  --config /content/qm9-sota/configs/train/smoke.yaml \
  --loss /content/qm9-sota/configs/loss/droplet_gentle.yaml \
  --run-name A1b_gentle_droplet
```

## Source-of-truth rule

- GitHub: source code, configs, documentation, small result summaries.
- Laptop: edit and commit code.
- Colab: disposable GPU runner.
- Google Drive: optional storage for checkpoints and large outputs.
