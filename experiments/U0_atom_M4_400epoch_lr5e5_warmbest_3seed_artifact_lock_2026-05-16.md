# U0_atom M4 400-epoch LR=5e-5 warm-best 3-seed artifact lock — 2026-05-16

## Facts

```text
target: U0_atom
target_index: 12
split: train 110000 / val 10000 / test 10831
metric: target-specific MAE
units: meV
conversion: raw PyG eV × 1000
model: plain M4 / pga_multivector_transformer
input information: atom/node graph features + graph edges + 3D positions + radial edge features + vector channels
method: fresh-optimizer LR=5e-5 refinement from each seed's raw best_model.pt
training window: resumed as epochs 301–400
hardware: NVIDIA A100-SXM4-40GB
warm-start git commit: ec4c3c15b982cfc2fbf0660101a718f2afcd6517
claim status: internal artifact-backed result only; not SOTA
```

## Raw 300-epoch M4 comparator

```text
seed42 raw test: 77.10036442513369 meV
seed43 raw test: 75.60250677898226 meV
seed44 raw test: 77.33998612544839 meV

raw mean test MAE: 76.68095244318812 meV
raw sample std: 0.9416147979926457 meV
```

## Warm-start results

| Seed | Warm val MAE | Warm test MAE | Raw test MAE | Test improvement |
|---:|---:|---:|---:|---:|
| 42 | 47.82959446310997 meV | 43.695705351478644 meV | 77.10036442513369 meV | 33.40465907365505 meV better |
| 43 | 45.70657138824463 meV | 48.63819113157208 meV | 75.60250677898226 meV | 26.96431564741018 meV better |
| 44 | 44.3827286362648 meV | 43.62081369747471 meV | 77.33998612544839 meV | 33.71917242797368 meV better |

## Warm-start aggregate

```text
warm mean val MAE: 45.97296482920647 meV
warm val sample std: 1.7388056550715703 meV

warm mean test MAE: 45.31823672684181 meV
warm test sample std: 2.8754086887042325 meV

mean test improvement over raw 300epoch M4: 31.362715716346308 meV
```

## Decisions

1. Promote the LR=5e-5 warm-best line to the current internal U0_atom production baseline.
2. Keep the raw 300-epoch M4 line as the baseline comparator.
3. Keep checkpoint averaging rejected; it worsened validation-selected test mean.
4. Do not make a SOTA claim until external reference tables are verified.

## Reproducibility status

```text
3 production seeds complete.
All three warm-start test eval JSONs are written.
Seed43 validation eval JSON is written.
Seed42 and seed44 validation values are currently from summary.json unless separate val eval JSONs are generated.
Artifacts are Drive-backed; checkpoints/results are not committed to GitHub.
```

## Open questions

1. Generate missing separate val eval JSONs for warm-start seeds 42 and 44.
2. Verify code diff between raw commit 30a26bd and warm-start commit ec4c3c15.
3. Test whether another lower-LR continuation window improves further.
4. Verify external SOTA comparison table, target definition, units, split, modality, and reproducibility status.

## Claim status

No SOTA claim is made here.

