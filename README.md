# hawkes

[![CI](https://github.com/OctopusTakopi/hawkes/actions/workflows/ci.yml/badge.svg)](https://github.com/OctopusTakopi/hawkes/actions/workflows/ci.yml)

`hawkes` is a Rust library for univariate and multivariate Hawkes processes with
sum-exponential kernels. It provides constrained fitting, allocation-free
online updates, seasonal baselines, marked likelihoods, diagnostics, and an
approximation for shifted power-law kernels.

Fitting uses **seconds**; the online API accepts **microsecond** timestamps.

## Model

For non-negative marks `v_i`, the sum-exponential conditional intensity is

```text
lambda(t) = mu * s(t) + sum_k alpha_k sum_{t_i < t} v_i exp(-beta_k (t - t_i)).
```

For homogeneous models, `s(t) = 1`. `PeriodicShape` provides a positive,
piecewise-constant seasonal profile normalized to mean one.

For independent marks with mean `E[V]`, stationarity requires

```text
E[V] * sum_k alpha_k / beta_k < 1.
```

Constructors and fitters enforce this constraint.

For `D` event components, the multivariate intensity is

```text
lambda_a(t) = mu_a * s_a(t)
            + sum_b sum_k alpha[a,b,k]
              sum_{i: component_i=b, t_i<t} v_i exp(-beta_k (t-t_i)).
```

The effective branching matrix is

```text
B[a,b] = E[V_b] * sum_k alpha[a,b,k] / beta_k.
```

`MultivariateSumExpHawkes` rejects models that fail a conservative spectral
radius bound. Its fitter constrains each source column's total offspring mass
below one, a sufficient condition for stationarity. Events with the same
timestamp form one batch and do not excite each other at zero lag.

## Installation

```toml
[dependencies]
hawkes = { git = "https://github.com/OctopusTakopi/hawkes", branch = "main" }
```

Pin a release tag or commit with `tag` or `rev` for reproducible builds.

## Online evaluation

```rust
use hawkes::{HawkesResult, SumExpHawkes};

fn main() -> HawkesResult<()> {
    // Decay time constants are 0.1 s and 1 s.
    let mut model = SumExpHawkes::new(
        0.5,
        vec![0.8, 0.1],
        vec![10.0, 1.0],
    )?;

    model.update(0, None)?;
    let future_excitation = model.evaluate(100_000)?; // 0.1 seconds later
    println!("future intensity = {}", model.mu() + future_excitation);
    Ok(())
}
```

- `update`: apply an event and return the resulting excitation
- `evaluate`: return future excitation without changing state
- `intensity`: return intensity after the latest update

## Fitting

Unmarked maximum likelihood with fixed decay rates:

```rust
use hawkes::{HawkesResult, SumExpHawkes};

fn fit(timestamps_seconds: Vec<f64>) -> HawkesResult<SumExpHawkes> {
    SumExpHawkes::fit(timestamps_seconds, vec![10.0, 1.0, 0.1])
}
```

For marked fitting, pass one non-negative mark per event and the expected mark
used in the stationarity constraint. Normalizing training marks to mean one is
usually convenient:

```rust
use hawkes::{HawkesResult, SumExpHawkes};

fn fit_marked(times: Vec<f64>, normalized_marks: Vec<f64>) -> HawkesResult<SumExpHawkes> {
    SumExpHawkes::fit_marked(times, normalized_marks, vec![10.0, 1.0], 1.0)
}
```

Decay rates are fixed; fitting estimates `mu` and `alpha_k`. Select decay scales
with chronological validation.

Fit caches use `N × K` space for univariate data and `N × D × K` for
multivariate data.

### Multivariate fitting

`MultivariateEvent` contains a timestamp, component index, and non-negative
mark. Combine duplicate component/timestamp records before fitting.

The seasonal fitter estimates `mu_a` and `alpha[a,b,k]` with fixed decay rates
and seasonal shapes.

`MultivariateFitConfig` controls the per-source offspring cap, minimum baseline
fraction, and iteration budget. `recalibrate_baselines` updates baseline rates
on an earlier calibration window without changing excitation parameters.

### Seasonal fitting

Estimate the seasonal shape on training data only, then fit its mean rate and
the Hawkes excitation jointly:

```rust
use hawkes::{HawkesResult, PeriodicShape, SumExpHawkes};

fn fit_daily(times: Vec<f64>, marks: Vec<f64>) -> HawkesResult<SumExpHawkes> {
    let shape = PeriodicShape::fit(&times, 86_400.0, 24, 60.0)?;
    SumExpHawkes::fit_marked_seasonal(times, marks, vec![100.0, 10.0, 1.0], 1.0, shape)
}
```

Timestamps set the profile phase. Use epoch seconds for a UTC time-of-day
profile.

### Conditional diagnostics

The `diagnostics` module scores from an explicit evaluation index; earlier
events only warm the model state. Reports include conditional log-likelihood,
compensator increments, a uniform KS diagnostic, and lag-one residual
autocorrelation. `TimeRescalingCriteria` applies configurable acceptance gates.

`AdaptiveTimingCalibrator` applies a causal Weibull renewal adjustment with
slow/fast score-driven intensity scales. `BernoulliScoreCalibrator` provides the
same predict-then-observe interface for binary marks.

## Examples and audits

- `cargo run --example basic_hawkes`
- `cargo run --example multi_scale`
- `cargo run --example power_law`
- `cargo run --release --example audit_ridgecrest`

[AUDIT.md](AUDIT.md) contains the frozen USGS Ridgecrest benchmark and its
time-rescaling checks.

## Scope and limitations

- Multivariate components share decay scales. The source-column stationarity
  constraint is conservative.
- Marks must be non-negative and are treated as exogenous; likelihoods are
  conditional on observed marks.
- Univariate timestamps must be strictly increasing. Aggregate ties first.
- Multivariate batches must be strictly increasing; put simultaneous events in
  one batch.
- The observation window runs from the first event to the last; no separate
  horizon parameter is available.
- Power-law kernels are approximated by a finite exponential basis. Generated
  approximations require `beta > 1` and must be stationary; validate accuracy
  over the relevant time range.
- `PeriodicShape::fit` is a two-stage marginal estimator, so clustering can leak
  into the seasonal profile.
- The `1.36 / sqrt(n)` KS threshold assumes a fully specified null. Prefer
  holdout diagnostics when parameters were estimated from the scored data.

## Quality and compatibility

CI checks formatting, Clippy, tests, documentation tests, and MSRV. The minimum
supported Rust version is 1.88.

Online sum-exponential updates allocate no heap memory after construction.
Benchmark them with `cargo bench --bench online_update`.

This software is provided for research and engineering use. It is not financial
advice and does not guarantee predictive performance.

## License

MIT. See [LICENSE](LICENSE).
