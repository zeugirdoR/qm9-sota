# SYS76 U0_atom motor-v9 CB screw coupling result - 2026-05-22

## Claim status

Internal artifact-backed validation result. Not a SOTA claim yet.

## Evaluation contract

target: U0_atom
target_index: 12
metric: target-specific validation MAE
unit: meV
split: train 110000 / val 10000 / test 10831

## Recovered mechanism

Recovered old MotorAttention used:

```text
q_screw / k_screw into 6D motor space
motor_score = asinh(G) * scale
gate = rbf_gate(rbf)
sig_motor = mean(abs(motor_score * gate))
reg_term = -sig_motor
```

Current v9 approximation:

```text
CB attention bias
6D screw motor attention
differentiable coupling reward
full data
batch_size: 256
lr: 3e-6
source checkpoint: SYS76_LOCAL_M4_CONTINUE_U0atom_400epoch_seed43
```

## Results

### plain400: SYS76_LOCAL_M4_CONTINUE_U0atom_400epoch_seed43

best_epoch: 400
best_val_target_norm_mae: 0.006369298789650202
best_val_target_converted_mae: 65.79087674617767 meV
git_commit_recorded_by_run: c308724091202a9d55925cb4aeb4d06f4430c57f

### plain600: SYS76_LOCAL_M4_CONTINUE_U0atom_600epoch_seed43

best_epoch: 598
best_val_target_norm_mae: 0.005605851765722036
best_val_target_converted_mae: 57.904910296201706 meV
git_commit_recorded_by_run: c308724091202a9d55925cb4aeb4d06f4430c57f

### plain1000: SYS76_LOCAL_M4_CONTINUE_U0atom_1000epoch_seed43

best_epoch: 992
best_val_target_norm_mae: 0.005035966169089079
best_val_target_converted_mae: 52.01833322644234 meV
git_commit_recorded_by_run: c308724091202a9d55925cb4aeb4d06f4430c57f

### plain1100: SYS76_LOCAL_M4_CONTINUE_U0atom_1100epoch_seed43

best_epoch: 1084
best_val_target_norm_mae: 0.0050138565711677074
best_val_target_converted_mae: 51.79000273346901 meV
git_commit_recorded_by_run: d67b7dfd83027e61e698564195cc1c40ea4c74da

### v9_smoke_or_half_batch128: SYS76_LOCAL_M4_MOTOR_V9_CB_SCREW_COUPLING_FROM400_U0atom_900epoch_seed43

best_epoch: 894
best_val_target_norm_mae: 0.005641045980155468
best_val_target_converted_mae: 58.26849117875099 meV
git_commit_recorded_by_run: d67b7dfd83027e61e698564195cc1c40ea4c74da

### v9_full_batch256_lr3e6: SYS76_LOCAL_M4_MOTOR_V9_FULL_BATCH256_LR3E6_FROM400_U0atom_900epoch_seed43

best_epoch: 899
best_val_target_norm_mae: 0.0047440072521567345
best_val_target_converted_mae: 49.00261387228966 meV
git_commit_recorded_by_run: d67b7dfd83027e61e698564195cc1c40ea4c74da

## Key comparison

```text
Plain M4-1100 seed43: 51.79000273346901 meV
Motor-v9 full batch256 seed43: 49.00261387228966 meV
delta vs plain1100: -2.787388861179352 meV

Plain M4-1000 seed43: 52.01833322644234 meV
delta vs plain1000: -3.0157193531526837 meV
```

## Decision

Promote motor-v9 full-data batch256 line for seed43 continuation and replication.

## Next

```text
1. Continue seed43 v9 best checkpoint to 1200 with lower LR.
2. Evaluate official val and test only after validation gate.
3. Replicate on seed42 and seed44.
4. Consider v10 contrastive negatives after locking v9.
```


## Seed43 v9 1200 continuation

Run:
SYS76_LOCAL_M4_MOTOR_V9_CONTINUE_FULL_BATCH256_LR1P5E6_U0atom_1200epoch_seed43

Result:
best_epoch: 1200
best_val_target_norm_mae: 0.004509382415562868
best_val_target_converted_mae: 46.579089015722275 meV
git_commit_recorded_by_run: 4196be6eef4448055050dffd98bd604485c06599

Comparison:
Plain M4-1100 seed43: 51.79000273346901 meV
v9 900 seed43:        49.00261387228966 meV
v9 1200 seed43:       46.579089015722275 meV

delta vs plain1100: -5.210913717746735 meV
delta vs v9 900:    -2.423524856567383 meV

Decision:
Continue seed43 v9 to 1500. Do not test yet because best_epoch is endpoint.
