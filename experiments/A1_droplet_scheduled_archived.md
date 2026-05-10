# A1 Droplet Scheduled: Archived Aggressive Schedule

## Result

Final mean normalized validation MAE: `0.374723`.

## Diagnostics

The gate did not collapse. Active fraction stayed near `1.0`.

By epoch 3:

- `lambda ≈ 0.739`
- `rho = nu * alpha ≈ 0.580`
- `temperature ≈ 18.73`

## Interpretation

The Droplet machinery was mechanically stable, but the schedule became too strong before the tiny model had stabilized.

## Decision

Archive this schedule. Next test: a gentler schedule with lower final lambda, lower final rho, lower final temperature, and a larger minimum weight.
