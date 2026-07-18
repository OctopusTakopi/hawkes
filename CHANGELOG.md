# Changelog

All notable changes to this project will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Conditional marked-Hawkes maximum-likelihood fitting.
- Unit-mean periodic baseline shapes and seasonal Hawkes fitting.
- Reusable conditional likelihood and time-rescaling diagnostics.
- Marked multivariate sum-exponential fitting, runtime timestamp batching,
  branching-matrix validation, and component-wise residual diagnostics.
- Configurable multivariate stationarity/baseline constraints, continuation
  fitting, convex baseline-only recalibration, and explicit residual gates.
- Serializable adaptive timing calibration with rolling Weibull renewal and
  score-driven intensity state, plus conditional Bernoulli score calibration.
- Reproducible Ridgecrest chronological audit.
- Public documentation, dataset provenance, and CI.

### Changed

- Model parameters and state are private and exposed through read-only getters.
- Fitting initialization scales with the empirical event rate.
- Fitting caches event recursions and compensator terms in one deterministic
  pass, avoiding exponentials during optimizer iterations and removing Rayon.
- Sum-exponential online updates use preallocated scratch state, evaluating
  each kernel decay once while retaining atomic failure semantics.
- Power-law approximation uses valid log-rate Laplace quadrature.

### Fixed

- Mark-aware stationarity and overflow-safe branching-ratio evaluation.
- Cached-likelihood input mutability and optimizer boundary rounding.
- Conditional likelihood now excludes the first event log-intensity, matching
  the documented conditioning convention.
- Weibull time-change likelihood corrections include both the hazard derivative
  and the transformed compensator difference.
