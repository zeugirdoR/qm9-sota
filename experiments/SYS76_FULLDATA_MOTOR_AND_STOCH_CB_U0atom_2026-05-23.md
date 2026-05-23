
## Stochastic CB robustness sweep update

Additional offline validation-only stochastic CB volume residual checks were run on:

- base checkpoint: SYS76_FULL_M4_MOTOR_V4A_CONTINUE_U0atom_2400epoch_batch256_seed43
- base validation MAE: approximately 47.2324 meV

Results:

| Variant | Best epoch | Best val MAE meV | Delta val meV |
|---|---:|---:|---:|
| minor2, samples4, hidden64 | 82 | 47.16499328613281 | -0.06740188598632812 |
| minor2, samples4, linear | 7 | 47.212738037109375 | -0.019672393798828125 |
| minor2, samples4, hidden16 | 93 | 47.21593475341797 | -0.01645660400390625 |
| minor2, samples8, hidden64 | 82 | 47.14659881591797 | -0.08578872680664062 |
| minor3, samples4, hidden64 | 94 | 47.20347213745117 | -0.02892303466796875 |

Interpretation:

- All stochastic CB variants improved validation.
- Linear residual also improved, so the volume summaries themselves carry residual signal.
- minor2 pair-volume summaries currently outperform minor3 triple-volume summaries.
- More stochastic samples helped: samples8 beat samples4.

Decision:

Promote stochastic CB positive-negative volume residual from one-off scout to online auxiliary-loss candidate.

Next implementation target:

loss = supervised_loss + lambda_cb_contrast * softplus(margin - volume_pos + volume_neg)

No test-set claim is made from these offline validation-only residual scouts.
