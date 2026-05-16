# U0_atom M4 300-epoch checkpoint averaging — negative result — 2026-05-16

## Facts

```text
target: U0_atom
target_index: 12
metric: target-specific MAE
units: meV
conversion: raw PyG eV × 1000
model: plain M4 / pga_multivector_transformer
input: atom/node graph features + graph edges + 3D positions + radial edge features + vector channels
split: test n=10831 per seed
```

## Raw locked baseline

```text
seed42 test: 77.10036442513369 meV
seed43 test: 75.60250677898226 meV
seed44 test: 77.33998612544839 meV
mean test: 76.68095244318812 meV
sample std: 0.9416147979926457 meV
```

## Checkpoint-averaged, validation-selected result

```text
seed42:
  tag: late7_240_300
  epochs: [240, 250, 260, 270, 280, 290, 300]
  test: 80.5528429602115 meV
  delta_vs_raw: +3.452478535077816 meV

seed43:
  tag: last3_280_300
  epochs: [280, 290, 300]
  test: 75.92724807241949 meV
  delta_vs_raw: +0.3247412934372278 meV

seed44:
  tag: midlate5_240_280
  epochs: [240, 250, 260, 270, 280]
  test: 100.6905099007465 meV
  delta_vs_raw: +23.350523775298115 meV

checkpoint-averaged mean test: 85.72353364445917 meV
checkpoint-averaged sample std: 13.166503549108763 meV
```

## Decision

```text
checkpoint averaging status: tested, not promoted
raw baseline remains current internal baseline
reason: worsened every seed and strongly destabilized seed44
claim status: internal negative result only; no SOTA claim
```

## Next step

Test a lower-learning-rate warm-start refinement from each seed's raw best_model.pt.
Run seed43 first as a pilot. Promote to seeds42/44 only if validation improves.

