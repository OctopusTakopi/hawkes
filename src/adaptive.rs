//! Predictable residual and binary-mark calibration for deployed point processes.

use crate::diagnostics::{
    ScoreDrivenScale, ScoreDrivenScaleConfig, TimeRescalingCriteria, WeibullTimeChange,
    diagnose_compensators,
};
use crate::{HawkesError, HawkesResult};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::VecDeque;

const SLOW_PERSISTENCE_GRID: [f64; 6] = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999];
const SLOW_STEP_GRID: [f64; 6] = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05];
const FAST_PERSISTENCE_GRID: [f64; 4] = [0.0, 0.5, 0.8, 0.9];
const FAST_STEP_GRID: [f64; 12] = [
    0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3,
];
const MAX_ABS_LOG_SCALE: f64 = 1.386_294_361_119_890_6;

/// Training and rolling-refit policy for [`AdaptiveTimingCalibrator`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AdaptiveTimingConfig {
    history_len: usize,
    refit_interval: usize,
    mean_tolerance: f64,
    max_ks_statistic: f64,
    max_abs_lag1_autocorrelation: f64,
    min_event_count: usize,
}

impl AdaptiveTimingConfig {
    /// Creates a validated adaptive timing policy.
    pub fn new(
        history_len: usize,
        refit_interval: usize,
        criteria: TimeRescalingCriteria,
    ) -> HawkesResult<Self> {
        if criteria.mean_tolerance() <= 0.0
            || criteria.max_ks_statistic() <= 0.0
            || criteria.max_abs_lag1_autocorrelation() <= 0.0
        {
            return Err(HawkesError::FittingError(
                "adaptive timing selection thresholds must be positive".to_string(),
            ));
        }
        let minimum_history = criteria.min_event_count().checked_mul(2).ok_or_else(|| {
            HawkesError::FittingError("adaptive timing history requirement overflowed".to_string())
        })?;
        if history_len < minimum_history {
            return Err(HawkesError::FittingError(
                "adaptive timing history must contain at least twice the diagnostic minimum"
                    .to_string(),
            ));
        }
        if refit_interval == 0 || refit_interval > history_len {
            return Err(HawkesError::FittingError(
                "adaptive timing refit interval must be in 1..=history_len".to_string(),
            ));
        }
        Ok(Self {
            history_len,
            refit_interval,
            mean_tolerance: criteria.mean_tolerance(),
            max_ks_statistic: criteria.max_ks_statistic(),
            max_abs_lag1_autocorrelation: criteria.max_abs_lag1_autocorrelation(),
            min_event_count: criteria.min_event_count(),
        })
    }

    /// Number of completed residuals retained for rolling fits.
    pub fn history_len(&self) -> usize {
        self.history_len
    }

    /// Number of new residuals between rolling refits.
    pub fn refit_interval(&self) -> usize {
        self.refit_interval
    }

    /// Residual effect-size criteria used for parameter selection.
    pub fn criteria(&self) -> TimeRescalingCriteria {
        TimeRescalingCriteria::new(
            self.mean_tolerance,
            self.max_ks_statistic,
            self.max_abs_lag1_autocorrelation,
            self.min_event_count,
        )
        .expect("AdaptiveTimingConfig stores validated criteria")
    }
}

impl Default for AdaptiveTimingConfig {
    fn default() -> Self {
        Self {
            history_len: 4_096,
            refit_interval: 512,
            mean_tolerance: 0.10,
            max_ks_statistic: 0.05,
            max_abs_lag1_autocorrelation: 0.05,
            min_event_count: 1_000,
        }
    }
}

impl<'de> Deserialize<'de> for AdaptiveTimingConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            history_len: usize,
            refit_interval: usize,
            mean_tolerance: f64,
            max_ks_statistic: f64,
            max_abs_lag1_autocorrelation: f64,
            min_event_count: usize,
        }
        let helper = Helper::deserialize(deserializer)?;
        let criteria = TimeRescalingCriteria::new(
            helper.mean_tolerance,
            helper.max_ks_statistic,
            helper.max_abs_lag1_autocorrelation,
            helper.min_event_count,
        )
        .map_err(serde::de::Error::custom)?;
        Self::new(helper.history_len, helper.refit_interval, criteria)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RollingWeibull {
    current: WeibullTimeChange,
    history: VecDeque<f64>,
    events_since_refit: usize,
}

impl RollingWeibull {
    fn observe(
        &mut self,
        base_increment: f64,
        config: AdaptiveTimingConfig,
    ) -> HawkesResult<crate::diagnostics::WeibullTimeChangeObservation> {
        let observation = self.current.observe(base_increment)?;
        if self.events_since_refit + 1 == config.refit_interval {
            let mut next_history = self.history.clone();
            push_bounded(&mut next_history, base_increment, config.history_len);
            let next = WeibullTimeChange::fit(next_history.make_contiguous())?;
            self.history = next_history;
            self.current = next;
            self.events_since_refit = 0;
        } else {
            push_bounded(&mut self.history, base_increment, config.history_len);
            self.events_since_refit += 1;
        }
        Ok(observation)
    }
}

/// Stateful, predictable time change fitted to base-model compensator increments.
///
/// The calibrator composes a rolling Weibull renewal hazard with slow and fast
/// score-driven intensity scales. Rolling refits use completed events only and
/// affect the next event, so the returned likelihood correction is causal.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AdaptiveTimingCalibrator {
    config: AdaptiveTimingConfig,
    renewal: RollingWeibull,
    slow: ScoreDrivenScale,
    fast: ScoreDrivenScale,
    fast_input_history: VecDeque<f64>,
    events_since_score_refit: usize,
}

/// One completed adaptive timing interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveTimingObservation {
    adjusted_increment: f64,
    log_likelihood_correction: f64,
    log_intensity_correction: f64,
}

/// Predictable adaptive timing quantities for a not-yet-observed event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveTimingPrediction {
    adjusted_increment: f64,
    log_likelihood_correction: f64,
    log_intensity_correction: f64,
}

impl AdaptiveTimingCalibrator {
    /// Fits the chronological calibration pipeline and initializes deployment state.
    pub fn fit(training_increments: &[f64], config: AdaptiveTimingConfig) -> HawkesResult<Self> {
        validate_history(training_increments)?;
        let minimum_training = config.history_len.checked_mul(2).ok_or_else(|| {
            HawkesError::FittingError("adaptive timing training requirement overflowed".to_string())
        })?;
        if training_increments.len() < minimum_training {
            return Err(HawkesError::FittingError(format!(
                "adaptive timing fit requires at least {} residuals",
                minimum_training
            )));
        }
        let split = training_increments.len() / 2;
        if split < config.history_len || training_increments.len() - split < config.history_len {
            return Err(HawkesError::FittingError(
                "adaptive timing chronological split is shorter than history_len".to_string(),
            ));
        }

        let initial =
            WeibullTimeChange::fit(&training_increments[split - config.history_len..split])?;
        let mut transformed = training_increments[..split]
            .iter()
            .map(|&increment| {
                initial
                    .observe(increment)
                    .map(|value| value.adjusted_increment())
            })
            .collect::<HawkesResult<Vec<_>>>()?;
        let mut selection_renewal = RollingWeibull {
            current: initial,
            history: training_increments[split - config.history_len..split]
                .iter()
                .copied()
                .collect(),
            events_since_refit: 0,
        };
        for &increment in &training_increments[split..] {
            transformed.push(
                selection_renewal
                    .observe(increment, config)?
                    .adjusted_increment(),
            );
        }

        let outer_renewal = WeibullTimeChange::fit(
            &training_increments[training_increments.len() - config.history_len..],
        )?;
        let max_fast_step = if outer_renewal.shape() >= 1.15 {
            0.05
        } else if outer_renewal.shape() >= 1.0 {
            0.1
        } else {
            0.2
        };
        let (slow_config, initial_fast_config) =
            select_two_scale(&transformed, config, max_fast_step)?;
        let mut slow = ScoreDrivenScale::new(slow_config);
        let mut fast = ScoreDrivenScale::new(initial_fast_config);
        let mut fast_input_history = VecDeque::with_capacity(config.history_len);
        for &increment in &transformed {
            let slow_observation = slow.observe(increment)?;
            let fast_input = slow_observation.adjusted_increment();
            fast.observe(fast_input)?;
            push_bounded(&mut fast_input_history, fast_input, config.history_len);
        }
        let fast_config = select_fast(fast_input_history.make_contiguous(), config)?;
        fast = ScoreDrivenScale::new(fast_config);
        for &increment in &fast_input_history {
            fast.observe(increment)?;
        }

        Ok(Self {
            config,
            renewal: RollingWeibull {
                current: outer_renewal,
                history: training_increments[training_increments.len() - config.history_len..]
                    .iter()
                    .copied()
                    .collect(),
                events_since_refit: 0,
            },
            slow,
            fast,
            fast_input_history,
            events_since_score_refit: 0,
        })
    }

    /// Applies the predictable time change and advances state after one event.
    pub fn observe(&mut self, base_increment: f64) -> HawkesResult<AdaptiveTimingObservation> {
        let renewal = self.renewal.observe(base_increment, self.config)?;
        let slow = self.slow.observe(renewal.adjusted_increment())?;
        let fast_input = slow.adjusted_increment();
        let fast = self.fast.observe(fast_input)?;
        let adjusted_increment = fast.adjusted_increment();
        let log_likelihood_correction = renewal.log_likelihood_correction()
            + slow.log_likelihood_correction()
            + fast.log_likelihood_correction();
        let log_intensity_correction =
            log_likelihood_correction + adjusted_increment - base_increment;

        push_bounded(
            &mut self.fast_input_history,
            fast_input,
            self.config.history_len,
        );
        self.events_since_score_refit += 1;
        if self.events_since_score_refit == self.config.refit_interval {
            let config = select_fast(self.fast_input_history.make_contiguous(), self.config)?;
            self.fast = ScoreDrivenScale::new(config);
            for &increment in &self.fast_input_history {
                self.fast.observe(increment)?;
            }
            self.events_since_score_refit = 0;
        }

        Ok(AdaptiveTimingObservation {
            adjusted_increment,
            log_likelihood_correction,
            log_intensity_correction,
        })
    }

    /// Predicts the time change at a candidate event without advancing state.
    ///
    /// `base_increment` is the base model's integrated intensity since the
    /// previous event. The exponential of `log_intensity_correction` multiplies
    /// the base conditional intensity at that candidate time.
    pub fn predict(&self, base_increment: f64) -> HawkesResult<AdaptiveTimingPrediction> {
        let renewal = self.renewal.current.observe(base_increment)?;
        let scale = self.slow.current_scale() * self.fast.current_scale();
        let adjusted_increment = scale * renewal.adjusted_increment();
        let renewal_log_intensity_correction =
            renewal.log_likelihood_correction() + renewal.adjusted_increment() - base_increment;
        let log_intensity_correction = renewal_log_intensity_correction + scale.ln();
        let log_likelihood_correction =
            log_intensity_correction - (adjusted_increment - base_increment);
        if !adjusted_increment.is_finite()
            || !log_intensity_correction.is_finite()
            || !log_likelihood_correction.is_finite()
        {
            return Err(HawkesError::InvalidCompensator);
        }
        Ok(AdaptiveTimingPrediction {
            adjusted_increment,
            log_likelihood_correction,
            log_intensity_correction,
        })
    }

    /// Active rolling Weibull time change.
    pub fn renewal_time_change(&self) -> WeibullTimeChange {
        self.renewal.current
    }

    /// Active slow score-driven scale configuration.
    pub fn slow_config(&self) -> ScoreDrivenScaleConfig {
        self.slow.config()
    }

    /// Active fast score-driven scale configuration.
    pub fn fast_config(&self) -> ScoreDrivenScaleConfig {
        self.fast.config()
    }

    /// Calibration and rolling-refit policy.
    pub fn config(&self) -> AdaptiveTimingConfig {
        self.config
    }
}

impl AdaptiveTimingObservation {
    /// Compensator increment after all predictable time changes.
    pub fn adjusted_increment(&self) -> f64 {
        self.adjusted_increment
    }

    /// Exact event-time log-likelihood change relative to the base model.
    pub fn log_likelihood_correction(&self) -> f64 {
        self.log_likelihood_correction
    }

    /// Log multiplier applied to base intensity at the completed event.
    pub fn log_intensity_correction(&self) -> f64 {
        self.log_intensity_correction
    }
}

impl AdaptiveTimingPrediction {
    /// Predicted calibrated compensator since the previous event.
    pub fn adjusted_increment(&self) -> f64 {
        self.adjusted_increment
    }

    /// Event-time log-likelihood correction if an event occurs at this point.
    pub fn log_likelihood_correction(&self) -> f64 {
        self.log_likelihood_correction
    }

    /// Log multiplier for base conditional intensity at this point.
    pub fn log_intensity_correction(&self) -> f64 {
        self.log_intensity_correction
    }
}

impl<'de> Deserialize<'de> for AdaptiveTimingCalibrator {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            config: AdaptiveTimingConfig,
            renewal: RollingWeibull,
            slow: ScoreDrivenScale,
            fast: ScoreDrivenScale,
            fast_input_history: VecDeque<f64>,
            events_since_score_refit: usize,
        }
        let helper = Helper::deserialize(deserializer)?;
        if helper.renewal.history.len() != helper.config.history_len
            || helper.fast_input_history.len() != helper.config.history_len
            || helper.renewal.events_since_refit >= helper.config.refit_interval
            || helper.events_since_score_refit >= helper.config.refit_interval
        {
            return Err(serde::de::Error::custom(
                "invalid adaptive timing rolling state",
            ));
        }
        validate_history(helper.renewal.history.as_slices().0)
            .and_then(|_| validate_history(helper.renewal.history.as_slices().1))
            .and_then(|_| validate_history(helper.fast_input_history.as_slices().0))
            .and_then(|_| validate_history(helper.fast_input_history.as_slices().1))
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            config: helper.config,
            renewal: helper.renewal,
            slow: helper.slow,
            fast: helper.fast,
            fast_input_history: helper.fast_input_history,
            events_since_score_refit: helper.events_since_score_refit,
        })
    }
}

/// Parameters for predictable score-driven calibration of a Bernoulli logit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BernoulliScoreConfig {
    persistence: f64,
    intercept_step: f64,
    slope_step: f64,
    max_abs_intercept: f64,
    min_slope: f64,
    max_slope: f64,
}

impl BernoulliScoreConfig {
    /// Creates a validated Bernoulli score-calibration configuration.
    pub fn new(
        persistence: f64,
        intercept_step: f64,
        slope_step: f64,
        max_abs_intercept: f64,
        min_slope: f64,
        max_slope: f64,
    ) -> HawkesResult<Self> {
        if !persistence.is_finite()
            || !(0.0..1.0).contains(&persistence)
            || !intercept_step.is_finite()
            || intercept_step < 0.0
            || !slope_step.is_finite()
            || slope_step < 0.0
            || !max_abs_intercept.is_finite()
            || max_abs_intercept <= 0.0
            || !min_slope.is_finite()
            || min_slope <= 0.0
            || !max_slope.is_finite()
            || max_slope < min_slope
            || !(min_slope..=max_slope).contains(&1.0)
        {
            return Err(HawkesError::FittingError(
                "invalid Bernoulli score-calibration configuration".to_string(),
            ));
        }
        Ok(Self {
            persistence,
            intercept_step,
            slope_step,
            max_abs_intercept,
            min_slope,
            max_slope,
        })
    }

    /// Autoregressive persistence of calibration state.
    pub fn persistence(&self) -> f64 {
        self.persistence
    }

    /// Intercept score step.
    pub fn intercept_step(&self) -> f64 {
        self.intercept_step
    }

    /// Slope score step.
    pub fn slope_step(&self) -> f64 {
        self.slope_step
    }

    /// Symmetric bound on the adaptive logit intercept.
    pub fn max_abs_intercept(&self) -> f64 {
        self.max_abs_intercept
    }

    /// Minimum adaptive logit slope.
    pub fn min_slope(&self) -> f64 {
        self.min_slope
    }

    /// Maximum adaptive logit slope.
    pub fn max_slope(&self) -> f64 {
        self.max_slope
    }
}

impl<'de> Deserialize<'de> for BernoulliScoreConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            persistence: f64,
            intercept_step: f64,
            slope_step: f64,
            max_abs_intercept: f64,
            min_slope: f64,
            max_slope: f64,
        }
        let helper = Helper::deserialize(deserializer)?;
        Self::new(
            helper.persistence,
            helper.intercept_step,
            helper.slope_step,
            helper.max_abs_intercept,
            helper.min_slope,
            helper.max_slope,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Predictable score-driven calibration state for a Bernoulli base logit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BernoulliScoreCalibrator {
    config: BernoulliScoreConfig,
    intercept: f64,
    slope: f64,
}

/// One calibrated Bernoulli observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BernoulliScoreObservation {
    probability: f64,
    log_likelihood_correction: f64,
}

impl BernoulliScoreCalibrator {
    /// Creates calibration state initialized to the identity map.
    pub fn new(config: BernoulliScoreConfig) -> Self {
        Self {
            config,
            intercept: 0.0,
            slope: 1.0,
        }
    }

    /// Returns the probability known before observing the outcome.
    pub fn predict(&self, base_logit: f64) -> HawkesResult<f64> {
        if !base_logit.is_finite() {
            return Err(HawkesError::FittingError(
                "Bernoulli base logit must be finite".to_string(),
            ));
        }
        Ok(logistic(self.intercept + self.slope * base_logit).clamp(1.0e-6, 1.0 - 1.0e-6))
    }

    /// Scores one outcome and updates calibration state for future outcomes.
    pub fn observe(
        &mut self,
        base_logit: f64,
        outcome: bool,
    ) -> HawkesResult<BernoulliScoreObservation> {
        let probability = self.predict(base_logit)?;
        let target = f64::from(outcome);
        let log_likelihood_correction = bernoulli_log_likelihood(probability, target)
            - bernoulli_log_likelihood_from_logit(base_logit, outcome);
        let score = target - probability;
        self.intercept =
            (self.config.persistence * self.intercept + self.config.intercept_step * score).clamp(
                -self.config.max_abs_intercept,
                self.config.max_abs_intercept,
            );
        self.slope = (1.0
            + self.config.persistence * (self.slope - 1.0)
            + self.config.slope_step * score * base_logit)
            .clamp(self.config.min_slope, self.config.max_slope);
        Ok(BernoulliScoreObservation {
            probability,
            log_likelihood_correction,
        })
    }

    /// Configuration governing this state.
    pub fn config(&self) -> BernoulliScoreConfig {
        self.config
    }

    /// Current adaptive intercept, known before the next outcome.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Current adaptive slope, known before the next outcome.
    pub fn slope(&self) -> f64 {
        self.slope
    }
}

impl BernoulliScoreObservation {
    /// Calibrated pre-outcome probability.
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Exact Bernoulli log-likelihood change relative to the base logit.
    pub fn log_likelihood_correction(&self) -> f64 {
        self.log_likelihood_correction
    }
}

impl<'de> Deserialize<'de> for BernoulliScoreCalibrator {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            config: BernoulliScoreConfig,
            intercept: f64,
            slope: f64,
        }
        let helper = Helper::deserialize(deserializer)?;
        if !helper.intercept.is_finite()
            || helper.intercept.abs() > helper.config.max_abs_intercept
            || !helper.slope.is_finite()
            || !(helper.config.min_slope..=helper.config.max_slope).contains(&helper.slope)
        {
            return Err(serde::de::Error::custom(
                "invalid Bernoulli score-calibration state",
            ));
        }
        Ok(Self {
            config: helper.config,
            intercept: helper.intercept,
            slope: helper.slope,
        })
    }
}

fn select_two_scale(
    increments: &[f64],
    config: AdaptiveTimingConfig,
    max_fast_step: f64,
) -> HawkesResult<(ScoreDrivenScaleConfig, ScoreDrivenScaleConfig)> {
    let evaluation_start = increments.len() / 2;
    let mut best = None;
    for slow_persistence in SLOW_PERSISTENCE_GRID {
        for slow_step in SLOW_STEP_GRID {
            for fast_persistence in FAST_PERSISTENCE_GRID {
                for fast_step in FAST_STEP_GRID
                    .into_iter()
                    .filter(|step| *step > 0.0 && *step <= max_fast_step)
                {
                    let slow_config = ScoreDrivenScaleConfig::new(
                        slow_persistence,
                        slow_step,
                        MAX_ABS_LOG_SCALE,
                    )?;
                    let fast_config = ScoreDrivenScaleConfig::new(
                        fast_persistence,
                        fast_step,
                        MAX_ABS_LOG_SCALE,
                    )?;
                    let mut slow = ScoreDrivenScale::new(slow_config);
                    let mut fast = ScoreDrivenScale::new(fast_config);
                    let mut correction = 0.0;
                    let mut adjusted = Vec::with_capacity(increments.len() - evaluation_start);
                    for (index, &increment) in increments.iter().enumerate() {
                        let slow_observation = slow.observe(increment)?;
                        let fast_observation =
                            fast.observe(slow_observation.adjusted_increment())?;
                        if index >= evaluation_start {
                            correction += slow_observation.log_likelihood_correction()
                                + fast_observation.log_likelihood_correction();
                            adjusted.push(fast_observation.adjusted_increment());
                        }
                    }
                    update_best(
                        &mut best,
                        &adjusted,
                        correction,
                        (slow_config, fast_config),
                        config,
                    )?;
                }
            }
        }
    }
    best.map(|(_, _, selected)| selected)
        .ok_or_else(|| HawkesError::FittingError("empty timing selection grid".to_string()))
}

fn select_fast(
    increments: &[f64],
    config: AdaptiveTimingConfig,
) -> HawkesResult<ScoreDrivenScaleConfig> {
    let evaluation_start = increments.len() / 2;
    let mut best = None;
    for persistence in FAST_PERSISTENCE_GRID {
        for score_step in FAST_STEP_GRID {
            let candidate =
                ScoreDrivenScaleConfig::new(persistence, score_step, MAX_ABS_LOG_SCALE)?;
            let mut scale = ScoreDrivenScale::new(candidate);
            let mut correction = 0.0;
            let mut adjusted = Vec::with_capacity(increments.len() - evaluation_start);
            for (index, &increment) in increments.iter().enumerate() {
                let observation = scale.observe(increment)?;
                if index >= evaluation_start {
                    correction += observation.log_likelihood_correction();
                    adjusted.push(observation.adjusted_increment());
                }
            }
            update_best(&mut best, &adjusted, correction, candidate, config)?;
        }
    }
    best.map(|(_, _, selected)| selected)
        .ok_or_else(|| HawkesError::FittingError("empty fast-score selection grid".to_string()))
}

fn update_best<T: Copy>(
    best: &mut Option<(f64, f64, T)>,
    adjusted: &[f64],
    correction: f64,
    candidate: T,
    config: AdaptiveTimingConfig,
) -> HawkesResult<()> {
    let report = diagnose_compensators(adjusted)?;
    let criteria = config.criteria();
    let loss = ((report.mean_compensator() - 1.0).abs() / criteria.mean_tolerance())
        .max(report.ks_statistic() / criteria.max_ks_statistic())
        .max(report.lag1_autocorrelation().abs() / criteria.max_abs_lag1_autocorrelation());
    if best.as_ref().is_none_or(|(best_loss, best_correction, _)| {
        loss < *best_loss || (loss == *best_loss && correction > *best_correction)
    }) {
        *best = Some((loss, correction, candidate));
    }
    Ok(())
}

fn validate_history(increments: &[f64]) -> HawkesResult<()> {
    if increments
        .iter()
        .any(|increment| !increment.is_finite() || *increment <= 0.0)
    {
        Err(HawkesError::InvalidCompensator)
    } else {
        Ok(())
    }
}

fn push_bounded(history: &mut VecDeque<f64>, value: f64, capacity: usize) {
    if history.len() == capacity {
        history.pop_front();
    }
    history.push_back(value);
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn bernoulli_log_likelihood(probability: f64, target: f64) -> f64 {
    target * probability.ln() + (1.0 - target) * (1.0 - probability).ln()
}

fn bernoulli_log_likelihood_from_logit(logit: f64, outcome: bool) -> f64 {
    if outcome {
        -softplus(-logit)
    } else {
        -softplus(logit)
    }
}

fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn training_residuals(count: usize) -> Vec<f64> {
        (0..count)
            .map(|index| {
                let probability = (index % 997 + 1) as f64 / 998.0;
                -probability.ln()
            })
            .collect()
    }

    #[test]
    fn adaptive_timing_is_causal_and_serializable() {
        let criteria = TimeRescalingCriteria::new(0.2, 0.2, 0.2, 20).unwrap();
        let config = AdaptiveTimingConfig::new(100, 25, criteria).unwrap();
        let training = training_residuals(400);
        let mut original = AdaptiveTimingCalibrator::fit(&training, config).unwrap();
        let prediction = original.predict(0.8).unwrap();
        let first = original.observe(0.8).unwrap();
        assert!((prediction.adjusted_increment() - first.adjusted_increment()).abs() < 1.0e-12);
        assert!(
            (prediction.log_likelihood_correction() - first.log_likelihood_correction()).abs()
                < 1.0e-12
        );
        assert!(
            (first.log_intensity_correction()
                - first.log_likelihood_correction()
                - first.adjusted_increment()
                + 0.8)
                .abs()
                < 1.0e-12
        );

        let encoded = serde_json::to_string(&original).unwrap();
        let mut restored: AdaptiveTimingCalibrator = serde_json::from_str(&encoded).unwrap();
        let left = original.observe(1.2).unwrap();
        let right = restored.observe(1.2).unwrap();
        assert!((left.adjusted_increment() - right.adjusted_increment()).abs() < 1.0e-12);
        assert!(
            (left.log_likelihood_correction() - right.log_likelihood_correction()).abs() < 1.0e-12
        );
    }

    #[test]
    fn adaptive_timing_rejects_short_history() {
        let config = AdaptiveTimingConfig::default();
        assert!(AdaptiveTimingCalibrator::fit(&[1.0; 100], config).is_err());
    }

    #[test]
    fn rolling_weibull_refit_failure_is_atomic() {
        let criteria = TimeRescalingCriteria::new(0.2, 0.2, 0.2, 2).unwrap();
        let config = AdaptiveTimingConfig::new(4, 1, criteria).unwrap();
        let mut rolling = RollingWeibull {
            current: WeibullTimeChange::new(1.0, 1.0).unwrap(),
            history: VecDeque::from(vec![1.0; 4]),
            events_since_refit: 0,
        };
        let before = rolling.clone();

        assert!(rolling.observe(1.0, config).is_err());
        assert_eq!(rolling, before);
    }

    #[test]
    fn bernoulli_calibration_updates_only_future_outcomes() {
        let config = BernoulliScoreConfig::new(0.9, 0.1, 0.05, 2.0, 0.25, 2.0).unwrap();
        let mut state = BernoulliScoreCalibrator::new(config);
        let first = state.observe(0.0, true).unwrap();
        assert_eq!(first.probability(), 0.5);
        assert!(state.predict(0.0).unwrap() > 0.5);

        let encoded = serde_json::to_string(&state).unwrap();
        let restored: BernoulliScoreCalibrator = serde_json::from_str(&encoded).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn bernoulli_likelihood_correction_is_finite_for_extreme_logits() {
        let config = BernoulliScoreConfig::new(0.9, 0.0, 0.0, 2.0, 0.25, 2.0).unwrap();
        let mut state = BernoulliScoreCalibrator::new(config);
        let observation = state.observe(1_000.0, false).unwrap();

        assert!(observation.log_likelihood_correction().is_finite());
        assert!(
            (observation.log_likelihood_correction() - (1_000.0 + 1.0e-6_f64.ln())).abs() < 1.0e-6
        );
    }
}
