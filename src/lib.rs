//! Validated univariate and multivariate Hawkes models, fitting, and diagnostics.
//!
//! Fitting timestamps are expressed in seconds. Online [`HawkesModel`] methods
//! accept microsecond timestamps. See the [crate README](https://github.com/OctopusTakopi/hawkes)
//! for model equations, marked fitting, examples, and limitations.
//!
//! # Quick start
//!
//! ```
//! use hawkes::{HawkesResult, SumExpHawkes};
//!
//! # fn main() -> HawkesResult<()> {
//! let mut model = SumExpHawkes::new(0.5, vec![0.8, 0.1], vec![10.0, 1.0])?;
//! model.update(0, None)?;
//! let excitation_after_100ms = model.evaluate(100_000)?;
//! assert!(excitation_after_100ms > 0.0);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

mod adaptive;
mod baseline;
pub mod diagnostics;
mod multivariate;
mod multivariate_fitting;
pub use adaptive::{
    AdaptiveTimingCalibrator, AdaptiveTimingConfig, AdaptiveTimingObservation,
    AdaptiveTimingPrediction, BernoulliScoreCalibrator, BernoulliScoreConfig,
    BernoulliScoreObservation,
};
pub use baseline::{Baseline, PeriodicShape};
pub use multivariate::{
    ComponentMark, MultivariateEvent, MultivariateFitConfig, MultivariateSumExpHawkes,
};

/// Errors returned when inputs, state, or fitted parameters violate model invariants.
#[derive(Error, Debug, PartialEq)]
#[non_exhaustive]
pub enum HawkesError {
    /// A named floating-point parameter was NaN or infinite.
    #[error("parameter {0} must be finite (got {1})")]
    NonFiniteParameter(&'static str, f64),
    /// The baseline intensity was negative.
    #[error("baseline intensity mu must be non-negative (got {0})")]
    InvalidMu(f64),
    /// An excitation coefficient was negative.
    #[error("excitation parameter alpha must be non-negative (got {0})")]
    InvalidAlpha(f64),
    /// A decay rate was not strictly positive.
    #[error("decay parameter beta must be positive (got {0})")]
    InvalidBeta(f64),
    /// The shifted power-law offset was not positive and finite.
    #[error("power-law shift delta must be positive and finite (got {0})")]
    InvalidDelta(f64),
    /// Alpha and beta vectors had different lengths.
    #[error("alphas number ({0}) and betas number ({1}) must be equal")]
    DimensionMismatch(usize, usize),
    /// A log-spaced approximation requested fewer than two scales.
    #[error("number of scales k must be at least 2 for log-spacing (got {0})")]
    InvalidLogSpacing(usize),
    /// The effective branching ratio was non-finite or not below one.
    #[error("branching ratio must be strictly less than 1 for stationarity (got {0})")]
    InvalidBranchingRatio(f64),
    /// An observed event volume was negative.
    #[error("event volume must be non-negative (got {0})")]
    InvalidVolume(f64),
    /// The configured expected event volume was negative.
    #[error("expected event volume must be non-negative (got {0})")]
    InvalidExpectedVolume(f64),
    /// A serialized excitation state was negative.
    #[error("excitation state must be non-negative (got {0})")]
    InvalidExcitation(f64),
    /// Serialized excitation state and kernel vectors had different lengths.
    #[error("excitations number ({0}) must match kernel count ({1})")]
    ExcitationDimensionMismatch(usize, usize),
    /// Marked fitting received a different number of volumes and timestamps.
    #[error("volumes number ({0}) must match timestamps number ({1})")]
    VolumeDimensionMismatch(usize, usize),
    /// Marked fitting requires a strictly positive expected volume.
    #[error("expected event volume must be positive for marked fitting (got {0})")]
    InvalidFittingExpectedVolume(f64),
    /// Fitting was requested without any events.
    #[error("timestamp list must contain at least one event")]
    EmptyTimestamps,
    /// Offline fitting timestamps were not sorted.
    #[error("timestamps must be sorted in nondecreasing order")]
    UnsortedTimestamps,
    /// Fitting timestamps did not span a positive finite interval.
    #[error("fitting observation window must have positive finite duration")]
    InvalidObservationWindow,
    /// An online update or evaluation preceded the most recent event.
    #[error("timestamp must be nondecreasing (previous {previous_us} us, got {current_us} us)")]
    NonMonotonicTimestamp {
        /// Most recent accepted event timestamp.
        previous_us: u64,
        /// Rejected update or evaluation timestamp.
        current_us: u64,
    },
    /// The numerical optimizer failed or returned invalid parameters.
    #[error("optimization failed: {0}")]
    FittingError(String),
    /// A periodic baseline period was not positive and finite.
    #[error("period must be positive and finite (got {0})")]
    InvalidPeriod(f64),
    /// A periodic baseline requested no bins.
    #[error("periodic baseline must contain at least one bin (got {0})")]
    InvalidBinCount(usize),
    /// Seasonal smoothing pseudo-exposure was not positive and finite.
    #[error("smoothing exposure must be positive and finite (got {0})")]
    InvalidSmoothing(f64),
    /// A periodic multiplier was non-positive or non-finite.
    #[error("seasonal multipliers must all be positive and finite")]
    InvalidSeasonalMultiplier,
    /// A baseline rate was not positive and finite.
    #[error("baseline rate must be positive and finite (got {0})")]
    InvalidBaselineRate(f64),
    /// A floating-point timestamp was NaN or infinite.
    #[error("timestamp must be finite (got {0})")]
    NonFiniteTimestamp(f64),
    /// An integration interval ended before it started.
    #[error("time interval must satisfy end >= start (got {start}..{end})")]
    InvalidTimeInterval {
        /// Interval start.
        start: f64,
        /// Interval end.
        end: f64,
    },
    /// Conditional scoring had no valid warm-up/test partition.
    #[error(
        "evaluation start {index} must be between 1 and event_count - 1 (event count {event_count})"
    )]
    InvalidEvaluationStart {
        /// First scored event index.
        index: usize,
        /// Total event count.
        event_count: usize,
    },
    /// A conditional intensity was non-positive or non-finite.
    #[error("conditional intensity must be positive and finite (got {0})")]
    InvalidConditionalIntensity(f64),
    /// No compensator increments were supplied for diagnostics.
    #[error("at least one compensator increment is required")]
    EmptyDiagnostics,
    /// A compensator increment was negative or non-finite.
    #[error("compensator increments must be non-negative and finite")]
    InvalidCompensator,
    /// A multivariate model requested no components.
    #[error("component count must be positive (got {0})")]
    InvalidComponentCount(usize),
    /// An event component was outside the model dimension.
    #[error("component {component} is outside component count {component_count}")]
    InvalidComponent {
        /// Invalid component index.
        component: usize,
        /// Number of model components.
        component_count: usize,
    },
    /// A flattened multivariate alpha tensor had the wrong length.
    #[error("alpha tensor length {actual} must equal D * D * K = {expected}")]
    InvalidAlphaTensor {
        /// Supplied flattened length.
        actual: usize,
        /// Required flattened length.
        expected: usize,
    },
    /// Per-component configuration had the wrong length.
    #[error("{name} length {actual} must equal component count {expected}")]
    ComponentDimensionMismatch {
        /// Configuration name.
        name: &'static str,
        /// Supplied length.
        actual: usize,
        /// Required component count.
        expected: usize,
    },
    /// Multivariate scoring attempted to split a timestamp batch.
    #[error("evaluation boundary must not split events with the same timestamp")]
    EvaluationSplitsBatch,
    /// A simple multivariate process received two events of one component at one time.
    #[error("component {component} occurs more than once at timestamp {timestamp}")]
    DuplicateComponentAtTimestamp {
        /// Duplicated component.
        component: usize,
        /// Shared timestamp in seconds.
        timestamp: f64,
    },
}

/// Result type returned by this crate.
pub type HawkesResult<T> = Result<T, HawkesError>;

mod fitting;

/// A trait representing a generic Hawkes Process model.
/// This allows for interchangeable use of different kernels (Exponential, SumExp, PowerLaw).
pub trait HawkesModel {
    /// Updates the state with a new event and returns the new excitation level.
    /// Volumes must have the expectation configured when the model was constructed.
    fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64>;
    /// Returns the theoretical excitation at `timestamp_us` without updating state.
    fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64>;
    /// Returns the current excitation level immediately after the last update.
    fn current_excitation(&self) -> f64;
    /// Returns the total intensity (lambda) = mu + excitation.
    fn intensity(&self) -> f64;
}

/// Computes the Hawkes excitation component iteratively in O(1) time.
/// Uses a single exponential kernel: g(t) = alpha * exp(-beta * t)
#[derive(Debug, Clone, Serialize)]
pub struct HawkesExcitation {
    mu: f64,
    alpha: f64,
    beta: f64,
    expected_volume: f64,
    current_excitation: f64,
    last_timestamp_us: Option<u64>,
}

impl HawkesExcitation {
    /// Creates a model for unmarked events or marks normalized to have mean 1.
    pub fn new(mu: f64, alpha: f64, beta: f64) -> HawkesResult<Self> {
        Self::new_with_expected_volume(mu, alpha, beta, 1.0)
    }

    /// Creates a marked model whose non-negative event volumes have the given expectation.
    pub fn new_with_expected_volume(
        mu: f64,
        alpha: f64,
        beta: f64,
        expected_volume: f64,
    ) -> HawkesResult<Self> {
        validate_nonnegative_finite("mu", mu)?;
        validate_nonnegative_finite("alpha", alpha)?;
        validate_nonnegative_finite("expected_volume", expected_volume)?;
        if !beta.is_finite() {
            return Err(HawkesError::NonFiniteParameter("beta", beta));
        }
        if beta <= 0.0 {
            return Err(HawkesError::InvalidBeta(beta));
        }
        validate_stationary_branching_ratio(scaled_branching_term(expected_volume, alpha, beta))?;
        Ok(Self {
            mu,
            alpha,
            beta,
            expected_volume,
            current_excitation: 0.0,
            last_timestamp_us: None,
        })
    }

    /// Applies an event whose volume follows the configured volume distribution.
    pub fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64> {
        let jump = self.alpha * validated_volume_factor(volume)?;

        let next_excitation = if let Some(last_time_us) = self.last_timestamp_us {
            let delta_t = checked_delta_t(last_time_us, timestamp_us)?;
            self.current_excitation * (-self.beta * delta_t).exp() + jump
        } else {
            jump
        };
        validate_nonnegative_finite("excitation", next_excitation)?;

        self.current_excitation = next_excitation;
        self.last_timestamp_us = Some(timestamp_us);
        Ok(self.current_excitation)
    }

    /// Returns excitation at a future microsecond timestamp without changing state.
    pub fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64> {
        if let Some(last_time_us) = self.last_timestamp_us {
            let delta_t = checked_delta_t(last_time_us, timestamp_us)?;
            Ok(self.current_excitation * (-self.beta * delta_t).exp())
        } else {
            Ok(0.0)
        }
    }

    /// Returns excitation immediately after the most recent update.
    pub fn current_excitation(&self) -> f64 {
        self.current_excitation
    }

    /// Returns the baseline intensity in events per second.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Returns the excitation jump coefficient.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the exponential decay rate in inverse seconds.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Returns the mark expectation used for stationarity validation.
    pub fn expected_volume(&self) -> f64 {
        self.expected_volume
    }

    /// Returns the most recent event timestamp, if any.
    pub fn last_timestamp_us(&self) -> Option<u64> {
        self.last_timestamp_us
    }

    /// Returns intensity immediately after the most recent update.
    pub fn intensity(&self) -> f64 {
        self.mu + self.current_excitation
    }

    /// Returns the effective marked branching ratio `E[V] * alpha / beta`.
    pub fn branching_ratio(&self) -> f64 {
        scaled_branching_term(self.expected_volume, self.alpha, self.beta)
    }
}

impl HawkesModel for HawkesExcitation {
    fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64> {
        self.update(timestamp_us, volume)
    }

    fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64> {
        self.evaluate(timestamp_us)
    }

    fn current_excitation(&self) -> f64 {
        self.current_excitation()
    }

    fn intensity(&self) -> f64 {
        self.intensity()
    }
}

/// Sum of Exponentials Hawkes process: g(t) = sum(alpha_i * exp(-beta_i * t))
/// Maintains O(K) complexity where K is the number of exponentials.
#[derive(Debug, Clone, Serialize)]
pub struct SumExpHawkes {
    mu: f64,
    alphas: Vec<f64>,
    betas: Vec<f64>,
    expected_volume: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    seasonal_shape: Option<PeriodicShape>,
    excitations: Vec<f64>,
    #[serde(skip)]
    next_excitations: Vec<f64>,
    last_timestamp_us: Option<u64>,
}

impl SumExpHawkes {
    /// Fits `mu` and `alpha_k` by maximum likelihood for an unmarked process.
    ///
    /// Timestamps must be sorted seconds and span a positive interval. `fixed_betas`
    /// are positive decay rates in inverse seconds and are not optimized.
    pub fn fit(timestamps: Vec<f64>, fixed_betas: Vec<f64>) -> HawkesResult<Self> {
        let (mu, alphas) = fitting::fit_hawkes(timestamps, fixed_betas.clone())?;
        Self::new(mu, alphas, fixed_betas)
    }

    /// Fits a marked Hawkes model conditional on observed non-negative volumes.
    ///
    /// Timestamps must be sorted seconds and have one corresponding volume each.
    /// `expected_volume` is the population expectation used in the stationarity
    /// condition. For marks normalized on training data, pass `1.0`. The mark
    /// density itself is not modeled.
    pub fn fit_marked(
        timestamps: Vec<f64>,
        volumes: Vec<f64>,
        fixed_betas: Vec<f64>,
        expected_volume: f64,
    ) -> HawkesResult<Self> {
        let (mu, alphas) =
            fitting::fit_marked_hawkes(timestamps, volumes, fixed_betas.clone(), expected_volume)?;
        Self::new_with_expected_volume(mu, alphas, fixed_betas, expected_volume)
    }

    /// Fits an unmarked Hawkes model with a fixed, unit-mean periodic baseline shape.
    ///
    /// The optimizer estimates the mean baseline rate `mu` and excitation
    /// coefficients. Fit the shape on training data only with [`PeriodicShape::fit`].
    pub fn fit_seasonal(
        timestamps: Vec<f64>,
        fixed_betas: Vec<f64>,
        seasonal_shape: PeriodicShape,
    ) -> HawkesResult<Self> {
        let (mu, alphas) =
            fitting::fit_seasonal_hawkes(timestamps, fixed_betas.clone(), &seasonal_shape)?;
        Self::new_with_seasonal_shape(mu, alphas, fixed_betas, 1.0, seasonal_shape)
    }

    /// Fits a conditional marked Hawkes model with a periodic baseline shape.
    pub fn fit_marked_seasonal(
        timestamps: Vec<f64>,
        volumes: Vec<f64>,
        fixed_betas: Vec<f64>,
        expected_volume: f64,
        seasonal_shape: PeriodicShape,
    ) -> HawkesResult<Self> {
        let (mu, alphas) = fitting::fit_marked_seasonal_hawkes(
            timestamps,
            volumes,
            fixed_betas.clone(),
            expected_volume,
            &seasonal_shape,
        )?;
        Self::new_with_seasonal_shape(mu, alphas, fixed_betas, expected_volume, seasonal_shape)
    }

    /// Creates a model for unmarked events or marks normalized to have mean 1.
    pub fn new(mu: f64, alphas: Vec<f64>, betas: Vec<f64>) -> HawkesResult<Self> {
        Self::new_with_expected_volume(mu, alphas, betas, 1.0)
    }

    /// Creates a marked model whose non-negative event volumes have the given expectation.
    pub fn new_with_expected_volume(
        mu: f64,
        alphas: Vec<f64>,
        betas: Vec<f64>,
        expected_volume: f64,
    ) -> HawkesResult<Self> {
        Self::new_validated(mu, alphas, betas, expected_volume, None)
    }

    /// Creates a marked model with a fixed, unit-mean periodic baseline shape.
    pub fn new_with_seasonal_shape(
        mu: f64,
        alphas: Vec<f64>,
        betas: Vec<f64>,
        expected_volume: f64,
        seasonal_shape: PeriodicShape,
    ) -> HawkesResult<Self> {
        Self::new_validated(mu, alphas, betas, expected_volume, Some(seasonal_shape))
    }

    fn new_validated(
        mu: f64,
        alphas: Vec<f64>,
        betas: Vec<f64>,
        expected_volume: f64,
        seasonal_shape: Option<PeriodicShape>,
    ) -> HawkesResult<Self> {
        validate_nonnegative_finite("mu", mu)?;
        validate_nonnegative_finite("expected_volume", expected_volume)?;
        if alphas.len() != betas.len() {
            return Err(HawkesError::DimensionMismatch(alphas.len(), betas.len()));
        }
        for &alpha in &alphas {
            validate_nonnegative_finite("alpha", alpha)?;
        }
        for &beta in &betas {
            if !beta.is_finite() {
                return Err(HawkesError::NonFiniteParameter("beta", beta));
            }
            if beta <= 0.0 {
                return Err(HawkesError::InvalidBeta(beta));
            }
        }
        validate_stationary_branching_ratio(marked_branching_ratio(
            expected_volume,
            &alphas,
            &betas,
        ))?;
        let k = alphas.len();
        Ok(Self {
            mu,
            alphas,
            betas,
            expected_volume,
            seasonal_shape,
            excitations: vec![0.0; k],
            next_excitations: vec![0.0; k],
            last_timestamp_us: None,
        })
    }

    /// Applies an event whose volume follows the configured volume distribution.
    pub fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64> {
        let volume_factor = validated_volume_factor(volume)?;

        let next_total = if let Some(last_time_us) = self.last_timestamp_us {
            let delta_t = checked_delta_t(last_time_us, timestamp_us)?;
            self.next_excitations
                .iter_mut()
                .zip(&self.excitations)
                .zip(&self.alphas)
                .zip(&self.betas)
                .map(|(((candidate, &excitation), &alpha), &beta)| {
                    *candidate = excitation * (-beta * delta_t).exp() + alpha * volume_factor;
                    *candidate
                })
                .sum()
        } else {
            self.next_excitations
                .iter_mut()
                .zip(&self.alphas)
                .map(|(candidate, &alpha)| {
                    *candidate = alpha * volume_factor;
                    *candidate
                })
                .sum()
        };
        validate_nonnegative_finite("excitation", next_total)?;

        self.excitations.copy_from_slice(&self.next_excitations);
        self.last_timestamp_us = Some(timestamp_us);
        Ok(next_total)
    }

    /// Returns total excitation at a future timestamp without changing state.
    pub fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64> {
        if let Some(last_time_us) = self.last_timestamp_us {
            let delta_t = checked_delta_t(last_time_us, timestamp_us)?;

            let excitation = self
                .excitations
                .iter()
                .zip(&self.betas)
                .map(|(&excitation, &beta)| excitation * (-beta * delta_t).exp())
                .sum();
            validate_nonnegative_finite("excitation", excitation)?;
            Ok(excitation)
        } else {
            Ok(0.0)
        }
    }

    /// Returns total excitation immediately after the most recent update.
    pub fn current_excitation(&self) -> f64 {
        self.excitations.iter().sum()
    }

    /// Returns the baseline intensity in events per second.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Returns the unit-mean periodic baseline shape, if configured.
    pub fn seasonal_shape(&self) -> Option<&PeriodicShape> {
        self.seasonal_shape.as_ref()
    }

    /// Returns baseline intensity at an offline timestamp measured in seconds.
    pub fn baseline_intensity(&self, timestamp_seconds: f64) -> HawkesResult<f64> {
        match &self.seasonal_shape {
            Some(shape) => Ok(self.mu * shape.multiplier(timestamp_seconds)?),
            None => {
                if !timestamp_seconds.is_finite() {
                    return Err(HawkesError::NonFiniteTimestamp(timestamp_seconds));
                }
                Ok(self.mu)
            }
        }
    }

    /// Integrates baseline intensity over an offline interval in seconds.
    pub fn integrated_baseline(&self, start: f64, end: f64) -> HawkesResult<f64> {
        match &self.seasonal_shape {
            Some(shape) => Ok(self.mu * shape.integrated_multiplier(start, end)?),
            None => {
                if !start.is_finite() {
                    return Err(HawkesError::NonFiniteTimestamp(start));
                }
                if !end.is_finite() {
                    return Err(HawkesError::NonFiniteTimestamp(end));
                }
                if end < start {
                    return Err(HawkesError::InvalidTimeInterval { start, end });
                }
                Ok(self.mu * (end - start))
            }
        }
    }

    /// Returns excitation coefficients in the same order as [`Self::betas`].
    pub fn alphas(&self) -> &[f64] {
        &self.alphas
    }

    /// Returns exponential decay rates in inverse seconds.
    pub fn betas(&self) -> &[f64] {
        &self.betas
    }

    /// Returns the mark expectation used for stationarity validation.
    pub fn expected_volume(&self) -> f64 {
        self.expected_volume
    }

    /// Returns per-kernel excitation state immediately after the latest event.
    pub fn excitations(&self) -> &[f64] {
        &self.excitations
    }

    /// Returns the most recent event timestamp, if any.
    pub fn last_timestamp_us(&self) -> Option<u64> {
        self.last_timestamp_us
    }

    /// Returns intensity immediately after the most recent update.
    pub fn intensity(&self) -> f64 {
        let baseline = self
            .last_timestamp_us
            .and_then(|timestamp| self.baseline_intensity(timestamp as f64 / 1_000_000.0).ok())
            .unwrap_or(self.mu);
        baseline + self.current_excitation()
    }

    /// Returns `E[V] * sum(alpha_k / beta_k)`.
    pub fn branching_ratio(&self) -> f64 {
        marked_branching_ratio(self.expected_volume, &self.alphas, &self.betas)
    }
}

impl HawkesModel for SumExpHawkes {
    fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64> {
        self.update(timestamp_us, volume)
    }

    fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64> {
        self.evaluate(timestamp_us)
    }

    fn current_excitation(&self) -> f64 {
        self.current_excitation()
    }

    fn intensity(&self) -> f64 {
        self.intensity()
    }
}

/// Approximate Power Law Hawkes process using a log-spaced sum of exponentials.
/// Power Law Kernel: g(t) = alpha / (delta + t)^beta
#[derive(Debug, Clone, Serialize)]
pub struct ApproxPowerLawHawkes {
    inner: SumExpHawkes,
}

impl ApproxPowerLawHawkes {
    /// Creates a log-spaced approximation with time scales from 1 ms to 100 s.
    pub fn new(mu: f64, alpha: f64, beta: f64, delta: f64, k: usize) -> HawkesResult<Self> {
        Self::new_with_expected_volume(mu, alpha, beta, delta, k, 1.0)
    }

    /// Creates a marked approximation whose non-negative event volumes have the given expectation.
    pub fn new_with_expected_volume(
        mu: f64,
        alpha: f64,
        beta: f64,
        delta: f64,
        k: usize,
        expected_volume: f64,
    ) -> HawkesResult<Self> {
        validate_nonnegative_finite("mu", mu)?;
        validate_nonnegative_finite("alpha", alpha)?;
        validate_nonnegative_finite("expected_volume", expected_volume)?;
        if !beta.is_finite() || beta <= 0.0 {
            return Err(if beta.is_finite() {
                HawkesError::InvalidBeta(beta)
            } else {
                HawkesError::NonFiniteParameter("beta", beta)
            });
        }
        if !delta.is_finite() || delta <= 0.0 {
            return Err(HawkesError::InvalidDelta(delta));
        }
        if k < 2 {
            return Err(HawkesError::InvalidLogSpacing(k));
        }

        let mut alphas = Vec::with_capacity(k);
        let mut betas = Vec::with_capacity(k);

        // Timescales span 0.001s to 100s, so decay rates span 0.01 to 1000.
        // The Laplace identity is
        // (delta + t)^(-beta) = ∫ b^(beta-1) exp(-b(delta + t)) db / Gamma(beta).
        // After substituting x = log(b), the quadrature weight is proportional to
        // b^beta exp(-delta*b) dx. Use trapezoidal half-weights at both endpoints.
        let log_spacing = 5.0 * std::f64::consts::LN_10 / (k - 1) as f64;
        let log_gamma_beta = ln_gamma_positive(beta);

        for i in 0..k {
            let b = 0.01 * 10.0f64.powf(i as f64 * 5.0 / (k - 1) as f64);
            let endpoint_weight: f64 = if i == 0 || i + 1 == k { 0.5 } else { 1.0 };
            let coefficient = if alpha == 0.0 {
                0.0
            } else {
                (alpha.ln() + log_spacing.ln() + endpoint_weight.ln() + beta * b.ln()
                    - delta * b
                    - log_gamma_beta)
                    .exp()
            };
            betas.push(b);
            alphas.push(coefficient);
        }

        Ok(Self {
            inner: SumExpHawkes::new_with_expected_volume(mu, alphas, betas, expected_volume)?,
        })
    }

    /// Creates an instance using manually specified scales (e.g. from offline fitting).
    pub fn with_scales(mu: f64, alphas: Vec<f64>, betas: Vec<f64>) -> HawkesResult<Self> {
        Self::with_scales_and_expected_volume(mu, alphas, betas, 1.0)
    }

    /// Creates a marked instance using manually specified scales.
    pub fn with_scales_and_expected_volume(
        mu: f64,
        alphas: Vec<f64>,
        betas: Vec<f64>,
        expected_volume: f64,
    ) -> HawkesResult<Self> {
        Ok(Self {
            inner: SumExpHawkes::new_with_expected_volume(mu, alphas, betas, expected_volume)?,
        })
    }

    /// Applies an event and returns total excitation immediately afterward.
    pub fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64> {
        self.inner.update(timestamp_us, volume)
    }

    /// Returns total excitation at a future timestamp without changing state.
    pub fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64> {
        self.inner.evaluate(timestamp_us)
    }

    /// Returns total excitation immediately after the most recent update.
    pub fn current_excitation(&self) -> f64 {
        self.inner.current_excitation()
    }

    /// Returns the baseline intensity in events per second.
    pub fn mu(&self) -> f64 {
        self.inner.mu()
    }

    /// Returns coefficients of the exponential approximation.
    pub fn alphas(&self) -> &[f64] {
        self.inner.alphas()
    }

    /// Returns decay rates of the exponential approximation.
    pub fn betas(&self) -> &[f64] {
        self.inner.betas()
    }

    /// Returns the mark expectation used for stationarity validation.
    pub fn expected_volume(&self) -> f64 {
        self.inner.expected_volume()
    }

    /// Returns the most recent event timestamp, if any.
    pub fn last_timestamp_us(&self) -> Option<u64> {
        self.inner.last_timestamp_us()
    }

    /// Returns intensity immediately after the most recent update.
    pub fn intensity(&self) -> f64 {
        self.inner.intensity()
    }

    /// Returns the effective branching ratio of the exponential approximation.
    pub fn branching_ratio(&self) -> f64 {
        self.inner.branching_ratio()
    }
}

fn validate_nonnegative_finite(name: &'static str, value: f64) -> HawkesResult<()> {
    if !value.is_finite() {
        return Err(HawkesError::NonFiniteParameter(name, value));
    }
    if value < 0.0 {
        return Err(match name {
            "mu" => HawkesError::InvalidMu(value),
            "alpha" => HawkesError::InvalidAlpha(value),
            "volume" => HawkesError::InvalidVolume(value),
            "expected_volume" => HawkesError::InvalidExpectedVolume(value),
            "excitation" => HawkesError::InvalidExcitation(value),
            _ => HawkesError::NonFiniteParameter(name, value),
        });
    }
    Ok(())
}

fn checked_delta_t(last_timestamp_us: u64, timestamp_us: u64) -> HawkesResult<f64> {
    if timestamp_us < last_timestamp_us {
        return Err(HawkesError::NonMonotonicTimestamp {
            previous_us: last_timestamp_us,
            current_us: timestamp_us,
        });
    }
    Ok((timestamp_us - last_timestamp_us) as f64 / 1_000_000.0)
}

fn marked_branching_ratio(expected_volume: f64, alphas: &[f64], betas: &[f64]) -> f64 {
    alphas
        .iter()
        .zip(betas.iter())
        .map(|(&alpha, &beta)| scaled_branching_term(expected_volume, alpha, beta))
        .sum()
}

fn scaled_branching_term(expected_volume: f64, alpha: f64, beta: f64) -> f64 {
    if expected_volume == 0.0 || alpha == 0.0 {
        return 0.0;
    }

    let unscaled = alpha / beta;
    if unscaled.is_finite() {
        expected_volume * unscaled
    } else {
        (expected_volume.ln() + alpha.ln() - beta.ln()).exp()
    }
}

fn ln_gamma_positive(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if value < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * value).sin().ln()
            - ln_gamma_positive(1.0 - value);
    }

    let shifted = value - 1.0;
    let series = COEFFICIENTS[1..]
        .iter()
        .enumerate()
        .fold(COEFFICIENTS[0], |sum, (index, coefficient)| {
            sum + coefficient / (shifted + index as f64 + 1.0)
        });
    let t = shifted + 7.5;

    0.5 * (2.0 * std::f64::consts::PI).ln() + (shifted + 0.5) * t.ln() - t + series.ln()
}

fn validate_stationary_branching_ratio(value: f64) -> HawkesResult<()> {
    if !value.is_finite() || value >= 1.0 {
        return Err(HawkesError::InvalidBranchingRatio(value));
    }
    Ok(())
}

fn validated_volume_factor(volume: Option<f64>) -> HawkesResult<f64> {
    let value = volume.unwrap_or(1.0);
    validate_nonnegative_finite("volume", value)?;
    Ok(value)
}

fn unit_expected_volume() -> f64 {
    1.0
}

impl<'de> Deserialize<'de> for HawkesExcitation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct HawkesExcitationHelper {
            mu: f64,
            alpha: f64,
            beta: f64,
            #[serde(default = "unit_expected_volume")]
            expected_volume: f64,
            current_excitation: f64,
            last_timestamp_us: Option<u64>,
        }

        let helper = HawkesExcitationHelper::deserialize(deserializer)?;
        let mut model = HawkesExcitation::new_with_expected_volume(
            helper.mu,
            helper.alpha,
            helper.beta,
            helper.expected_volume,
        )
        .map_err(serde::de::Error::custom)?;
        validate_nonnegative_finite("excitation", helper.current_excitation)
            .map_err(serde::de::Error::custom)?;
        model.current_excitation = helper.current_excitation;
        model.last_timestamp_us = helper.last_timestamp_us;
        Ok(model)
    }
}

impl<'de> Deserialize<'de> for SumExpHawkes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SumExpHawkesHelper {
            mu: f64,
            alphas: Vec<f64>,
            betas: Vec<f64>,
            #[serde(default = "unit_expected_volume")]
            expected_volume: f64,
            #[serde(default)]
            seasonal_shape: Option<PeriodicShape>,
            excitations: Vec<f64>,
            last_timestamp_us: Option<u64>,
        }

        let helper = SumExpHawkesHelper::deserialize(deserializer)?;
        if helper.excitations.len() != helper.alphas.len() {
            return Err(serde::de::Error::custom(
                HawkesError::ExcitationDimensionMismatch(
                    helper.excitations.len(),
                    helper.alphas.len(),
                ),
            ));
        }

        for &excitation in &helper.excitations {
            validate_nonnegative_finite("excitation", excitation)
                .map_err(serde::de::Error::custom)?;
        }
        validate_nonnegative_finite("excitation", helper.excitations.iter().sum())
            .map_err(serde::de::Error::custom)?;

        let mut model = SumExpHawkes::new_validated(
            helper.mu,
            helper.alphas,
            helper.betas,
            helper.expected_volume,
            helper.seasonal_shape,
        )
        .map_err(serde::de::Error::custom)?;
        model.excitations = helper.excitations;
        model.last_timestamp_us = helper.last_timestamp_us;
        Ok(model)
    }
}

impl<'de> Deserialize<'de> for ApproxPowerLawHawkes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ApproxPowerLawHawkesHelper {
            inner: SumExpHawkes,
        }

        let helper = ApproxPowerLawHawkesHelper::deserialize(deserializer)?;
        Ok(Self {
            inner: helper.inner,
        })
    }
}

impl HawkesModel for ApproxPowerLawHawkes {
    fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64> {
        self.update(timestamp_us, volume)
    }

    fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64> {
        self.evaluate(timestamp_us)
    }

    fn current_excitation(&self) -> f64 {
        self.current_excitation()
    }

    fn intensity(&self) -> f64 {
        self.intensity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_vs_update() {
        let mut hawkes = HawkesExcitation::new(0.5, 0.8, 1.0).unwrap();

        hawkes.update(0, None).unwrap();
        assert_eq!(hawkes.current_excitation(), 0.8);
        assert_eq!(hawkes.intensity(), 1.3); // mu + excitation

        let eval_val = hawkes.evaluate(1_000_000).unwrap();
        assert!((eval_val - 0.2943).abs() < 0.01);

        assert_eq!(hawkes.current_excitation(), 0.8);

        hawkes.update(1_000_000, None).unwrap();
        let cur = hawkes.current_excitation();
        assert!((cur - 1.0943).abs() < 0.01);
    }

    #[test]
    fn test_serialization() {
        let mut hawkes = HawkesExcitation::new_with_expected_volume(0.5, 0.8, 1.0, 0.5).unwrap();
        hawkes.update(0, None).unwrap();

        // Serialize
        let json = serde_json::to_string(&hawkes).unwrap();

        // Deserialize
        let loaded: HawkesExcitation = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.current_excitation(), hawkes.current_excitation());
        assert_eq!(loaded.mu(), 0.5);
        assert_eq!(loaded.expected_volume(), 0.5);
    }

    #[test]
    fn test_sum_exp_scratch_state_is_not_serialized() {
        let mut model = SumExpHawkes::new(0.5, vec![0.2, 0.1], vec![1.0, 2.0]).unwrap();
        model.update(0, None).unwrap();
        model.update(100_000, Some(0.5)).unwrap();

        let json = serde_json::to_string(&model).unwrap();
        assert!(!json.contains("next_excitations"));
        let mut loaded: SumExpHawkes = serde_json::from_str(&json).unwrap();

        let expected = model.update(200_000, Some(1.5)).unwrap();
        let actual = loaded.update(200_000, Some(1.5)).unwrap();
        assert!((actual - expected).abs() < 1.0e-12);
        assert_eq!(loaded.excitations(), model.excitations());
    }

    #[test]
    fn test_rejects_invalid_model_parameters() {
        assert!(matches!(
            HawkesExcitation::new(-0.1, 1.0, 1.0),
            Err(HawkesError::InvalidMu(_))
        ));
        assert!(matches!(
            HawkesExcitation::new(0.1, -1.0, 1.0),
            Err(HawkesError::InvalidAlpha(_))
        ));
        assert!(matches!(
            HawkesExcitation::new(0.1, 1.0, f64::NAN),
            Err(HawkesError::NonFiniteParameter("beta", _))
        ));
        assert!(matches!(
            SumExpHawkes::new(0.1, vec![1.0, -0.1], vec![1.0, 2.0]),
            Err(HawkesError::InvalidAlpha(_))
        ));
        assert!(matches!(
            ApproxPowerLawHawkes::new(0.1, 1.0, 1.0, 0.0, 5),
            Err(HawkesError::InvalidDelta(_))
        ));
        assert!(matches!(
            HawkesExcitation::new(0.1, 1.0, 1.0),
            Err(HawkesError::InvalidBranchingRatio(_))
        ));
        assert!(matches!(
            SumExpHawkes::new(0.1, vec![0.6, 0.5], vec![1.0, 1.0]),
            Err(HawkesError::InvalidBranchingRatio(_))
        ));
    }

    #[test]
    fn test_rejects_invalid_volume_inputs() {
        let mut single = HawkesExcitation::new(0.5, 0.8, 1.0).unwrap();
        assert!(matches!(
            single.update(0, Some(-1.0)),
            Err(HawkesError::InvalidVolume(_))
        ));
        assert!(matches!(
            single.update(0, Some(f64::NAN)),
            Err(HawkesError::NonFiniteParameter("volume", _))
        ));

        let mut multi = SumExpHawkes::new(0.1, vec![0.2, 0.1], vec![1.0, 2.0]).unwrap();
        assert!(matches!(
            multi.update(0, Some(-1.0)),
            Err(HawkesError::InvalidVolume(_))
        ));
    }

    #[test]
    fn test_update_overflow_does_not_mutate_state() {
        let mut single =
            HawkesExcitation::new_with_expected_volume(0.1, f64::MAX, 1.0, 0.0).unwrap();
        assert!(matches!(
            single.update(1, Some(2.0)),
            Err(HawkesError::NonFiniteParameter("excitation", _))
        ));
        assert_eq!(single.current_excitation(), 0.0);
        assert_eq!(single.last_timestamp_us(), None);

        let mut multi = SumExpHawkes::new_with_expected_volume(
            0.1,
            vec![f64::MAX, f64::MAX],
            vec![1.0, 1.0],
            0.0,
        )
        .unwrap();
        assert!(matches!(
            multi.update(1, Some(1.0)),
            Err(HawkesError::NonFiniteParameter("excitation", _))
        ));
        assert_eq!(multi.excitations(), &[0.0, 0.0]);
        assert_eq!(multi.last_timestamp_us(), None);
    }

    #[test]
    fn test_marked_stationarity_uses_expected_volume() {
        assert!(matches!(
            HawkesExcitation::new_with_expected_volume(0.1, 0.4, 1.0, 3.0),
            Err(HawkesError::InvalidBranchingRatio(ratio)) if (ratio - 1.2).abs() < 1.0e-12
        ));
        assert!(matches!(
            SumExpHawkes::new_with_expected_volume(
                0.1,
                vec![0.2, 0.1],
                vec![1.0, 1.0],
                4.0,
            ),
            Err(HawkesError::InvalidBranchingRatio(ratio)) if (ratio - 1.2).abs() < 1.0e-12
        ));

        let model = HawkesExcitation::new_with_expected_volume(0.1, 1.2, 1.0, 0.5).unwrap();
        assert!((model.branching_ratio() - 0.6).abs() < 1.0e-12);
        assert_eq!(model.expected_volume(), 0.5);
    }

    #[test]
    fn test_marked_branching_ratio_scales_before_summing() {
        let model = SumExpHawkes::new_with_expected_volume(
            0.1,
            vec![1.0e308, 1.0e308],
            vec![1.0, 1.0],
            1.0e-310,
        )
        .unwrap();

        assert!((model.branching_ratio() - 0.02).abs() < 1.0e-12);

        let zero_mark_model = SumExpHawkes::new_with_expected_volume(
            0.1,
            vec![f64::MAX],
            vec![f64::MIN_POSITIVE],
            0.0,
        )
        .unwrap();
        assert_eq!(zero_mark_model.branching_ratio(), 0.0);

        let minimum_positive = f64::from_bits(1);
        let scaled_overflow = HawkesExcitation::new_with_expected_volume(
            0.1,
            0.5,
            minimum_positive,
            minimum_positive,
        )
        .unwrap();
        assert!((scaled_overflow.branching_ratio() - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn test_rejects_nonfinite_branching_ratio() {
        assert!(matches!(
            SumExpHawkes::new_with_expected_volume(
                0.1,
                vec![f64::MAX],
                vec![f64::MIN_POSITIVE],
                1.0,
            ),
            Err(HawkesError::InvalidBranchingRatio(ratio)) if !ratio.is_finite()
        ));
    }

    #[test]
    fn test_power_law_quadrature_matches_kernel_at_origin() {
        let mut model = ApproxPowerLawHawkes::new(0.1, 0.05, 1.5, 0.01, 100).unwrap();
        let approximated = model.update(0, None).unwrap();
        let expected = 0.05 / 0.01f64.powf(1.5);

        assert!((approximated - expected).abs() / expected < 5.0e-4);
    }

    #[test]
    fn test_public_marked_fit_returns_stationary_model() {
        let timestamps = (0..40).map(|index| index as f64 * 0.1).collect();
        let volumes = (0..40)
            .map(|index| if index % 2 == 0 { 0.5 } else { 1.5 })
            .collect();
        let model = SumExpHawkes::fit_marked(timestamps, volumes, vec![10.0, 1.0], 1.0).unwrap();

        assert!(model.branching_ratio().is_finite());
        assert!(model.branching_ratio() < 1.0);
        assert_eq!(model.expected_volume(), 1.0);
    }

    #[test]
    fn test_public_seasonal_fit_and_serialization() {
        let timestamps = (0..40).map(|index| index as f64 * 0.1).collect::<Vec<_>>();
        let shape = PeriodicShape::new(2.0, vec![0.5, 1.5]).unwrap();
        let model = SumExpHawkes::fit_seasonal(timestamps, vec![10.0, 1.0], shape).unwrap();
        let encoded = serde_json::to_string(&model).unwrap();
        let decoded: SumExpHawkes = serde_json::from_str(&encoded).unwrap();

        assert!(decoded.seasonal_shape().is_some());
        assert_eq!(decoded.seasonal_shape(), model.seasonal_shape());
        assert!(decoded.branching_ratio() < 1.0);
    }

    #[test]
    fn test_public_marked_fit_preserves_input_errors() {
        assert!(matches!(
            SumExpHawkes::fit_marked(vec![0.0, 1.0], vec![1.0], vec![1.0], 1.0),
            Err(HawkesError::VolumeDimensionMismatch(1, 2))
        ));
    }

    #[test]
    fn test_deserialization_validates_state() {
        let unstable = r#"{"mu":0.5,"alpha":1.0,"beta":1.0,"current_excitation":0.0,"last_timestamp_us":null}"#;
        assert!(serde_json::from_str::<HawkesExcitation>(unstable).is_err());

        let mismatched = r#"{
            "mu":0.5,
            "alphas":[0.2,0.1],
            "betas":[1.0,2.0],
            "excitations":[0.1],
            "last_timestamp_us":null
        }"#;
        assert!(serde_json::from_str::<SumExpHawkes>(mismatched).is_err());
    }

    #[test]
    fn test_rejects_out_of_order_online_timestamps() {
        let mut hawkes = HawkesExcitation::new(0.5, 0.8, 1.0).unwrap();
        hawkes.update(1_000_000, None).unwrap();

        assert!(matches!(
            hawkes.update(999_999, None),
            Err(HawkesError::NonMonotonicTimestamp {
                previous_us: 1_000_000,
                current_us: 999_999
            })
        ));
        assert!(matches!(
            hawkes.evaluate(999_999),
            Err(HawkesError::NonMonotonicTimestamp {
                previous_us: 1_000_000,
                current_us: 999_999
            })
        ));
    }
}
