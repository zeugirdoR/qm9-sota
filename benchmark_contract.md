# QM9-SOTA Benchmark Contract

## Dataset

Canonical dataset for the current prototype:

- PyTorch Geometric QM9
- Preprocessed PyG path
- No RDKit processing for initial runs
- Molecules: 130,831
- Targets: 19
- 3D coordinates included

## Split

Seed: `42`

- Train: first 110,000 shuffled indices
- Validation: next 10,000 shuffled indices
- Test: remaining 10,831 shuffled indices

## Smoke mode

Used for fast Colab debugging:

- Smoke train: first 20,000 training molecules
- Smoke validation: first 2,000 validation molecules
- Batch size: 128

## Target normalization

- Compute target mean/std from the full training split only.
- Model predicts normalized targets.
- Primary smoke metric: mean normalized MAE across 19 targets.
- Secondary metric: raw per-target MAE.

## Canonical runtime

- Google Colab GPU runtime
- Dataset cache: `/content/data/QM9`
- Code source: GitHub repo
- Colab notebook role: clone/install/run only

## Known result history

### A0 baseline, 3 epochs

Mean normalized validation MAE: `0.349324`

### A1 aggressive scheduled Droplet, 3 epochs

Mean normalized validation MAE: `0.374723`

Interpretation: mechanically stable, but schedule became too aggressive by epoch 3.

### A0 paired baseline, 5 epochs

Mean normalized validation MAE: `0.322820`

This is the current smoke-test baseline to beat.
