# hawkes

[![CI](https://github.com/OctopusTakopi/hawkes/actions/workflows/ci.yml/badge.svg)](https://github.com/OctopusTakopi/hawkes/actions/workflows/ci.yml)

`hawkes` is a Rust library for validated univariate and multivariate Hawkes processes. It
supports constant-time exponential updates, multi-scale sum-exponential models,
stationarity-constrained maximum likelihood, conditional marked likelihoods,
periodic piecewise-constant baselines, causal residual calibration,
time-rescaling diagnostics, and a Laplace-quadrature approximation to shifted
power-law kernels.

The crate treats time in **seconds** during fitting and accepts **microsecond**
timestamps in the online API.

## Model

For non-negative marks `v_i`, the sum-exponential conditional intensity is

```text
lambda(t) = mu * s(t) + sum_k alpha_k sum_{t_i < t} v_i exp(-beta_k (t - t_i)).
```

Here `s(t) = 1` for the homogeneous model. A fitted `PeriodicShape` is
positive, piecewise constant, periodic, and normalized to mean one, so `mu`
remains the mean baseline rate over a complete period.

For independent marks with mean `E[V]`, stationarity requires

```text
E[V] * sum_k alpha_k / beta_k < 1.
```

Constructors validate this invariant. Fitting uses a constrained
parameterization with representable slack below one.

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

`MultivariateSumExpHawkes` validates a conservative upper bound on the spectral
radius of `B`. Its fitter constrains every source column's total offspring mass
below one, which is sufficient for stationarity. Equal-timestamp components are
treated as one batch and cannot excite one another at zero lag.

## Installation

```toml
[dependencies]
hawkes = { git = "https://github.com/OctopusTakopi/hawkes", branch = "main" }
```

For reproducible builds, pin a release tag or commit with `tag` or `rev`.

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

`update` returns excitation immediately after the event. `evaluate` returns
future excitation without changing state. `intensity` returns intensity
immediately after the most recent update.

## Fitting

Unmarked maximum likelihood with fixed decay rates:

```rust
use hawkes::{HawkesResult, SumExpHawkes};

fn fit(timestamps_seconds: Vec<f64>) -> HawkesResult<SumExpHawkes> {
    SumExpHawkes::fit(timestamps_seconds, vec![10.0, 1.0, 0.1])
}
```

For marked fitting, pass one non-negative mark per event and the mark
expectation used by the stationarity condition. Normalizing marks on training
data to mean one is usually numerically convenient:

```rust
use hawkes::{HawkesResult, SumExpHawkes};

fn fit_marked(times: Vec<f64>, normalized_marks: Vec<f64>) -> HawkesResult<SumExpHawkes> {
    SumExpHawkes::fit_marked(times, normalized_marks, vec![10.0, 1.0], 1.0)
}
```

Decay rates are fixed hyperparameters; this crate currently estimates `mu` and
`alpha_k` only. Use chronological validation to select decay scales.

Fitting precomputes parameter-independent event histories and compensator terms
once. This avoids repeated exponential recursion during optimization. The cache
is `N × K` for univariate fits and `N × D × K` for multivariate fits.

### Multivariate fitting

`MultivariateEvent` carries an absolute timestamp, component index, and
non-negative mark. The seasonal fitter estimates one `mu_a` and all
`alpha[a,b,k]` values while decay rates and train-only seasonal shapes remain
fixed. Input must contain at most one event per component and timestamp; combine
duplicate records before fitting.

`MultivariateFitConfig` sets a strict per-source offspring cap, a minimum
baseline fraction, and the optimizer iteration budget. Positive baseline floors
use a zero-floor continuation fit before the constrained solve to avoid poor
logit corners. `recalibrate_baselines` can update only baseline rates on a
strictly earlier calibration window while leaving excitation and stationarity
unchanged.

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

The timestamps establish the profile phase. Use absolute epoch seconds when a
UTC time-of-day profile is intended.

### Conditional diagnostics

The reusable `diagnostics` module scores only events at or after an explicit
evaluation index. Earlier events warm the excitation recursion. Reports include
conditional log-likelihood, compensator increments, their mean, a uniform KS
diagnostic, and lag-one residual autocorrelation. `TimeRescalingCriteria`
applies explicit mean, KS-effect, autocorrelation, and sample-size gates.

`AdaptiveTimingCalibrator` fits a chronological Weibull renewal time change and
slow/fast score-driven intensity scales to base-model compensators. Rolling
refits use completed events only. `predict` supplies a calibrated compensator
and intensity multiplier without changing state; `observe` advances state and
returns the exact likelihood correction. `BernoulliScoreCalibrator` provides
the analogous predictable correction for a conditional binary mark. Both
runtime states validate on deserialization.

## Examples and audits

- `cargo run --example basic_hawkes`
- `cargo run --example multi_scale`
- `cargo run --example power_law`
- `cargo run --release --example audit_ridgecrest`

[AUDIT.md](AUDIT.md) records the frozen USGS Ridgecrest benchmark. It shows that
the model improves conditional likelihood over Poisson while also demonstrating
why time-rescaling calibration must be checked.

## Scope and limitations

- Multivariate models share decay scales across components. The sufficient
  source-column constraint is more conservative than unconstrained spectral-radius fitting.
- Marks must be non-negative. Their distribution is treated as exogenous; the
  likelihood is conditional on observed marks.
- Online timestamps must be nondecreasing. Exact ties are accepted by the API,
  but applications modeling a simple point process should aggregate tied events.
- Fitting conditions on the first observed event and assumes sorted timestamps.
- The approximate power-law model uses a finite range of exponential scales and
  is an approximation, not an exact power-law state representation.

## Quality and compatibility

The repository CI runs formatting, Clippy with warnings denied, all tests,
documentation tests, and MSRV checks. The minimum supported Rust version is
1.88.

The online sum-exponential update uses preallocated scratch state and performs
no heap allocation after model construction. Run its throughput smoke benchmark
with `cargo bench --bench online_update`.

This software is provided for research and engineering use. It is not financial
advice and does not guarantee predictive performance.

## License

MIT. See [LICENSE](LICENSE).
