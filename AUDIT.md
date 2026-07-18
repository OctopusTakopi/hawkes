# Model audits

## 2019 Ridgecrest earthquake sequence

This audit uses 2,062 magnitude 2.5+ records from the frozen USGS Ridgecrest
catalog in `data/ridgecrest_2019_m2_5.csv`. One exact timestamp tie is deduplicated,
leaving 2,061 timestamp batches. The split is chronological: the first 80% of
batches are used for fitting and the remaining 20% for conditional one-step-ahead
scoring. No holdout events are used to estimate parameters.

Run the reproducible audit with:

```sh
cargo run --release --example audit_ridgecrest
```

## Setup

- Train/test timestamp batches: 1,648 / 413
- Total observation window: 7.28 days
- Fixed exponential decay scales: 5 minutes, 1 hour, and 1 day
- Baseline: homogeneous Poisson rate estimated on the training window
- Calibration diagnostic: time-rescaled holdout compensator increments
- Audit date: 2026-07-18

## Results

| Metric | Hawkes | Poisson / target |
|---|---:|---:|
| Holdout log-likelihood | -3926.629 | -4643.391 |
| Predictive lift | +1.735502 nats/event | 0 |
| Branching ratio | 0.99999999 | strictly below 1 |
| Mean rescaled increment | 0.997281 | 1.0 |
| Uniform KS statistic | 0.233584 | 0.066921 at 5% |
| Lag-one residual correlation | -0.306464 | near 0 |
| Release-mode fit time | under 1 ms in this audit | hardware-dependent |

The model substantially outperforms a constant-rate Poisson process on
conditional holdout likelihood. The gain is approximately 1.74 nats per event.

The executable audit reports `REJECT`. Its branching ratio reaches the
constrained upper boundary, the time-rescaling KS statistic exceeds the 5%
critical value, and the residuals have substantial lag-one dependence despite
a mean compensator close to one. Thus this dataset verifies that the
implementation learns useful short-term clustering, but it rejects the current
stationary sum-exponential specification as a complete earthquake model.

Likely causes include the strongly nonstationary aftershock rate, magnitude-
dependent triggering, catalog incompleteness immediately after the main shocks,
and fixed rather than estimated decay scales. Reasonable next model extensions
would be an Omori/power-law aftershock kernel, magnitude marks, and a
time-varying background intensity.

The fitter now initializes `mu` from the empirical event rate. Before that
change, this seconds-scale dataset caused the line search to stall because the
fixed initial value of 1 event/second was several orders of magnitude above the
catalog rate.
