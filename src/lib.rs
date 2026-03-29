//! # hawkes
//!
//! A high-performance Hawkes Process library for modeling and fitting self-exciting point processes.
//!
//! ## Overview
//! A Hawkes process is a type of point process where the arrival of an event increases the probability
//! of future events. This library provides tools for:
//! - **Modeling**: Single or multi-scale (Sum of Exponentials) kernels.
//! - **Fitting**: Estimating parameters from historical data using Maximum Likelihood Estimation (MLE).
//! - **Real-time Evaluation**: Efficient $O(1)$ and $O(K)$ updates.

use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HawkesError {
    #[error("parameter {0} must be finite (got {1})")]
    NonFiniteParameter(&'static str, f64),
    #[error("baseline intensity mu must be non-negative (got {0})")]
    InvalidMu(f64),
    #[error("excitation parameter alpha must be non-negative (got {0})")]
    InvalidAlpha(f64),
    #[error("decay parameter beta must be positive (got {0})")]
    InvalidBeta(f64),
    #[error("power-law shift delta must be positive and finite (got {0})")]
    InvalidDelta(f64),
    #[error("alphas number ({0}) and betas number ({1}) must be equal")]
    DimensionMismatch(usize, usize),
    #[error("number of scales k must be at least 2 for log-spacing (got {0})")]
    InvalidLogSpacing(usize),
    #[error("branching ratio must be strictly less than 1 for stationarity (got {0})")]
    InvalidBranchingRatio(f64),
    #[error("event volume must be non-negative (got {0})")]
    InvalidVolume(f64),
    #[error("excitation state must be non-negative (got {0})")]
    InvalidExcitation(f64),
    #[error("excitations number ({0}) must match kernel count ({1})")]
    ExcitationDimensionMismatch(usize, usize),
    #[error("timestamp list must contain at least one event")]
    EmptyTimestamps,
    #[error("timestamps must be sorted in nondecreasing order")]
    UnsortedTimestamps,
    #[error("timestamp must be nondecreasing (previous {previous_us} us, got {current_us} us)")]
    NonMonotonicTimestamp { previous_us: u64, current_us: u64 },
    #[error("optimization failed: {0}")]
    FittingError(String),
}

/// Start with a Result type for better error handling.
pub type HawkesResult<T> = Result<T, HawkesError>;

pub mod fitting;

/// A trait representing a generic Hawkes Process model.
/// This allows for interchangeable use of different kernels (Exponential, SumExp, PowerLaw).
pub trait HawkesModel {
    /// Updates the state with a new event and returns the new excitation level.
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
    pub mu: f64,
    pub alpha: f64,
    pub beta: f64,
    pub current_excitation: f64,
    pub last_timestamp_us: Option<u64>,
}

impl HawkesExcitation {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> HawkesResult<Self> {
        validate_nonnegative_finite("mu", mu)?;
        validate_nonnegative_finite("alpha", alpha)?;
        if !beta.is_finite() {
            return Err(HawkesError::NonFiniteParameter("beta", beta));
        }
        if beta <= 0.0 {
            return Err(HawkesError::InvalidBeta(beta));
        }
        validate_stationary_branching_ratio(alpha / beta)?;
        Ok(Self {
            mu,
            alpha,
            beta,
            current_excitation: 0.0,
            last_timestamp_us: None,
        })
    }

    pub fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64> {
        let jump = self.alpha * validated_volume_factor(volume)?;

        if let Some(last_time_us) = self.last_timestamp_us {
            let delta_t = checked_delta_t(last_time_us, timestamp_us)?;
            self.current_excitation = self.current_excitation * (-self.beta * delta_t).exp() + jump;
        } else {
            self.current_excitation = jump;
        }

        self.last_timestamp_us = Some(timestamp_us);
        Ok(self.current_excitation)
    }

    pub fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64> {
        if let Some(last_time_us) = self.last_timestamp_us {
            let delta_t = checked_delta_t(last_time_us, timestamp_us)?;
            Ok(self.current_excitation * (-self.beta * delta_t).exp())
        } else {
            Ok(0.0)
        }
    }

    pub fn current_excitation(&self) -> f64 {
        self.current_excitation
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }

    pub fn last_timestamp_us(&self) -> Option<u64> {
        self.last_timestamp_us
    }

    pub fn intensity(&self) -> f64 {
        // Note: strictly speaking intensity() is usually asked at "current time".
        // If we want intensity immediately after update:
        self.mu + self.current_excitation
    }

    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
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
    pub mu: f64,
    pub alphas: Vec<f64>,
    pub betas: Vec<f64>,
    pub excitations: Vec<f64>,
    pub last_timestamp_us: Option<u64>,
}

impl SumExpHawkes {
    /// Fits the model parameters (mu, alphas) to the given timestamps using Maximum Likelihood Estimation.
    /// The `betas` are fixed hyperparameters (timescales).
    pub fn fit(timestamps: Vec<f64>, fixed_betas: Vec<f64>) -> HawkesResult<Self> {
        let (mu, alphas) = fitting::fit_hawkes(timestamps, fixed_betas.clone())
            .map_err(|e| HawkesError::FittingError(e.to_string()))?;
        Self::new(mu, alphas, fixed_betas)
    }

    pub fn new(mu: f64, alphas: Vec<f64>, betas: Vec<f64>) -> HawkesResult<Self> {
        validate_nonnegative_finite("mu", mu)?;
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
        validate_stationary_branching_ratio(branching_ratio(&alphas, &betas))?;
        let k = alphas.len();
        Ok(Self {
            mu,
            alphas,
            betas,
            excitations: vec![0.0; k],
            last_timestamp_us: None,
        })
    }

    pub fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64> {
        let volume_factor = validated_volume_factor(volume)?;

        if let Some(last_time_us) = self.last_timestamp_us {
            let delta_t = checked_delta_t(last_time_us, timestamp_us)?;

            // Optimized using iterators instead of index lookups
            let iter = self
                .excitations
                .iter_mut()
                .zip(self.alphas.iter())
                .zip(self.betas.iter());

            for ((excitation, &alpha), &beta) in iter {
                let jump = alpha * volume_factor;
                *excitation = *excitation * (-beta * delta_t).exp() + jump;
            }
        } else {
            for (excitation, &alpha) in self.excitations.iter_mut().zip(self.alphas.iter()) {
                *excitation = alpha * volume_factor;
            }
        }

        self.last_timestamp_us = Some(timestamp_us);
        Ok(self.current_excitation())
    }

    pub fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64> {
        if let Some(last_time_us) = self.last_timestamp_us {
            let delta_t = checked_delta_t(last_time_us, timestamp_us)?;

            Ok(self
                .excitations
                .iter()
                .zip(self.betas.iter())
                .map(|(&excitation, &beta)| excitation * (-beta * delta_t).exp())
                .sum())
        } else {
            Ok(0.0)
        }
    }

    pub fn current_excitation(&self) -> f64 {
        self.excitations.iter().sum()
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn alphas(&self) -> &[f64] {
        &self.alphas
    }

    pub fn betas(&self) -> &[f64] {
        &self.betas
    }

    pub fn excitations(&self) -> &[f64] {
        &self.excitations
    }

    pub fn last_timestamp_us(&self) -> Option<u64> {
        self.last_timestamp_us
    }

    pub fn intensity(&self) -> f64 {
        self.mu + self.current_excitation()
    }

    pub fn branching_ratio(&self) -> f64 {
        branching_ratio(&self.alphas, &self.betas)
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
    /// Creates a new default log-spaced approximation.
    pub fn new(mu: f64, alpha: f64, beta: f64, delta: f64, k: usize) -> HawkesResult<Self> {
        validate_nonnegative_finite("mu", mu)?;
        validate_nonnegative_finite("alpha", alpha)?;
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

        // Log-spacing width: betas span from 0.001 to 100 (5 decades in log10 = ln(10^5)).
        // Each of the (k-1) intervals covers Δlog(β) = ln(10^5)/(k-1) in natural log space.
        // This is the correct quadrature weight for the log-spaced approximation: instead of
        // a uniform 1/k (which assumes linear spacing), we must use the actual log-spacing width
        // so that slower (wider) timescale bins contribute their proper share.
        let log_spacing = 5.0 * std::f64::consts::LN_10 / (k - 1) as f64;

        for i in 0..k {
            // Log-spaced timescales from 0.001s to 100s
            let b = 0.001 * 10.0f64.powf(i as f64 * 5.0 / (k - 1) as f64);
            betas.push(b);
            alphas.push(alpha * log_spacing / (delta + 1.0 / b).powf(beta));
        }

        Ok(Self {
            inner: SumExpHawkes::new(mu, alphas, betas)?,
        })
    }

    /// Creates an instance using manually specified scales (e.g. from offline fitting).
    pub fn with_scales(mu: f64, alphas: Vec<f64>, betas: Vec<f64>) -> HawkesResult<Self> {
        Ok(Self {
            inner: SumExpHawkes::new(mu, alphas, betas)?,
        })
    }

    pub fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64> {
        self.inner.update(timestamp_us, volume)
    }

    pub fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64> {
        self.inner.evaluate(timestamp_us)
    }

    pub fn current_excitation(&self) -> f64 {
        self.inner.current_excitation()
    }

    pub fn mu(&self) -> f64 {
        self.inner.mu()
    }

    pub fn alphas(&self) -> &[f64] {
        self.inner.alphas()
    }

    pub fn betas(&self) -> &[f64] {
        self.inner.betas()
    }

    pub fn last_timestamp_us(&self) -> Option<u64> {
        self.inner.last_timestamp_us()
    }

    pub fn intensity(&self) -> f64 {
        self.inner.intensity()
    }

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

fn branching_ratio(alphas: &[f64], betas: &[f64]) -> f64 {
    alphas
        .iter()
        .zip(betas.iter())
        .map(|(&alpha, &beta)| alpha / beta)
        .sum()
}

fn validate_stationary_branching_ratio(value: f64) -> HawkesResult<()> {
    if value >= 1.0 {
        return Err(HawkesError::InvalidBranchingRatio(value));
    }
    Ok(())
}

fn validated_volume_factor(volume: Option<f64>) -> HawkesResult<f64> {
    let value = volume.unwrap_or(1.0);
    validate_nonnegative_finite("volume", value)?;
    Ok(value)
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
            current_excitation: f64,
            last_timestamp_us: Option<u64>,
        }

        let helper = HawkesExcitationHelper::deserialize(deserializer)?;
        let mut model = HawkesExcitation::new(helper.mu, helper.alpha, helper.beta)
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

        let mut model = SumExpHawkes::new(helper.mu, helper.alphas, helper.betas)
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
        let mut hawkes = HawkesExcitation::new(0.5, 0.8, 1.0).unwrap();
        hawkes.update(0, None).unwrap();

        // Serialize
        let json = serde_json::to_string(&hawkes).unwrap();

        // Deserialize
        let loaded: HawkesExcitation = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.current_excitation(), hawkes.current_excitation());
        assert_eq!(loaded.mu(), 0.5);
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
