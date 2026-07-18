//! Conditional likelihood and time-rescaling diagnostics.

use crate::multivariate::{alpha_index, validate_events};
use crate::{
    Baseline, HawkesError, HawkesResult, MultivariateEvent, MultivariateSumExpHawkes, SumExpHawkes,
};
use serde::{Deserialize, Deserializer, Serialize};

/// Time-rescaling diagnostics for conditional inter-event compensators.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeRescalingReport {
    event_count: usize,
    mean_compensator: f64,
    ks_statistic: f64,
    ks_critical_value_5pct: f64,
    lag1_autocorrelation: f64,
}

/// Effect-size gates for deciding whether residual calibration is operationally acceptable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeRescalingCriteria {
    mean_tolerance: f64,
    max_ks_statistic: f64,
    max_abs_lag1_autocorrelation: f64,
    min_event_count: usize,
}

/// Parameters for a bounded score-driven multiplicative intensity scale.
///
/// If `z` is the base-model compensator increment and `c` is the scale known
/// before the event, the calibrated increment is `c * z`. After observing the
/// event, the log scale follows
/// `x_next = persistence * x + score_step * (1 - c * z)`, clipped to the
/// configured bound. This is a predictable observation-driven model: the
/// current event affects only future intensity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ScoreDrivenScaleConfig {
    persistence: f64,
    score_step: f64,
    max_abs_log_scale: f64,
}

impl ScoreDrivenScaleConfig {
    /// Creates a validated score-driven scale configuration.
    pub fn new(persistence: f64, score_step: f64, max_abs_log_scale: f64) -> HawkesResult<Self> {
        if !persistence.is_finite() || !(0.0..1.0).contains(&persistence) {
            return Err(HawkesError::FittingError(
                "score-driven persistence must be finite and in [0, 1)".to_string(),
            ));
        }
        if !score_step.is_finite() || score_step < 0.0 {
            return Err(HawkesError::FittingError(
                "score-driven step must be finite and non-negative".to_string(),
            ));
        }
        if !max_abs_log_scale.is_finite() || max_abs_log_scale <= 0.0 {
            return Err(HawkesError::FittingError(
                "score-driven log-scale bound must be finite and positive".to_string(),
            ));
        }
        Ok(Self {
            persistence,
            score_step,
            max_abs_log_scale,
        })
    }

    /// Autoregressive persistence of the log intensity scale.
    pub fn persistence(&self) -> f64 {
        self.persistence
    }

    /// Step applied to the conditional likelihood score.
    pub fn score_step(&self) -> f64 {
        self.score_step
    }

    /// Symmetric bound on the absolute log intensity scale.
    pub fn max_abs_log_scale(&self) -> f64 {
        self.max_abs_log_scale
    }
}

impl<'de> Deserialize<'de> for ScoreDrivenScaleConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            persistence: f64,
            score_step: f64,
            max_abs_log_scale: f64,
        }
        let helper = Helper::deserialize(deserializer)?;
        Self::new(
            helper.persistence,
            helper.score_step,
            helper.max_abs_log_scale,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// One predictable score-driven intensity-scale state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ScoreDrivenScale {
    config: ScoreDrivenScaleConfig,
    log_scale: f64,
}

/// Result of observing one base-model compensator increment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreDrivenObservation {
    adjusted_increment: f64,
    log_likelihood_correction: f64,
}

/// Weibull renewal time-change applied to a base compensator interval.
///
/// For base increment `z`, the transformed cumulative hazard is
/// `(z / scale)^shape`. Its instantaneous hazard is the derivative of this
/// transformation times the base intensity, so the likelihood correction is
/// exact and predictable within each component interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct WeibullTimeChange {
    shape: f64,
    scale: f64,
}

/// One transformed Weibull-renewal compensator interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullTimeChangeObservation {
    adjusted_increment: f64,
    log_likelihood_correction: f64,
}

impl WeibullTimeChange {
    /// Creates a validated Weibull time-change.
    pub fn new(shape: f64, scale: f64) -> HawkesResult<Self> {
        if !shape.is_finite() || shape <= 0.0 {
            return Err(HawkesError::FittingError(
                "Weibull shape must be finite and positive".to_string(),
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(HawkesError::FittingError(
                "Weibull scale must be finite and positive".to_string(),
            ));
        }
        Ok(Self { shape, scale })
    }

    /// Fits Weibull shape and scale by maximum likelihood.
    pub fn fit(increments: &[f64]) -> HawkesResult<Self> {
        if increments.len() < 2
            || increments
                .iter()
                .any(|&increment| !increment.is_finite() || increment <= 0.0)
        {
            return Err(HawkesError::InvalidCompensator);
        }
        let logs = increments
            .iter()
            .map(|increment| increment.ln())
            .collect::<Vec<_>>();
        let mean_log = logs.iter().sum::<f64>() / logs.len() as f64;
        let equation = |shape: f64| {
            let max_scaled = logs
                .iter()
                .map(|&log| shape * log)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut weight_sum = 0.0;
            let mut weighted_log_sum = 0.0;
            for &log in &logs {
                let weight = (shape * log - max_scaled).exp();
                weight_sum += weight;
                weighted_log_sum += weight * log;
            }
            1.0 / shape + mean_log - weighted_log_sum / weight_sum
        };
        let mut lower = 0.05;
        let mut upper = 20.0;
        if equation(lower) <= 0.0 || equation(upper) >= 0.0 {
            return Err(HawkesError::FittingError(
                "could not bracket Weibull shape".to_string(),
            ));
        }
        for _ in 0..100 {
            let midpoint = 0.5 * (lower + upper);
            if equation(midpoint) > 0.0 {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
        }
        let shape = 0.5 * (lower + upper);
        let max_scaled = logs
            .iter()
            .map(|&log| shape * log)
            .fold(f64::NEG_INFINITY, f64::max);
        let scaled_sum = logs
            .iter()
            .map(|&log| (shape * log - max_scaled).exp())
            .sum::<f64>();
        let log_scale = (max_scaled + (scaled_sum / increments.len() as f64).ln()) / shape;
        Self::new(shape, log_scale.exp())
    }

    /// Weibull shape parameter.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Weibull scale parameter in base-compensator units.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Transforms one positive base compensator interval.
    pub fn observe(&self, base_increment: f64) -> HawkesResult<WeibullTimeChangeObservation> {
        if !base_increment.is_finite() || base_increment <= 0.0 {
            return Err(HawkesError::InvalidCompensator);
        }
        let log_increment = base_increment.ln();
        let log_scale = self.scale.ln();
        let adjusted_increment = (self.shape * (log_increment - log_scale)).exp();
        let log_hazard =
            self.shape.ln() - self.shape * log_scale + (self.shape - 1.0) * log_increment;
        let correction = log_hazard - (adjusted_increment - base_increment);
        if !adjusted_increment.is_finite() || !correction.is_finite() {
            return Err(HawkesError::InvalidCompensator);
        }
        Ok(WeibullTimeChangeObservation {
            adjusted_increment,
            log_likelihood_correction: correction,
        })
    }
}

impl<'de> Deserialize<'de> for WeibullTimeChange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            shape: f64,
            scale: f64,
        }
        let helper = Helper::deserialize(deserializer)?;
        Self::new(helper.shape, helper.scale).map_err(serde::de::Error::custom)
    }
}

impl WeibullTimeChangeObservation {
    /// Time-rescaled increment after applying the renewal transformation.
    pub fn adjusted_increment(&self) -> f64 {
        self.adjusted_increment
    }

    /// Exact conditional log-likelihood change relative to the base intensity.
    pub fn log_likelihood_correction(&self) -> f64 {
        self.log_likelihood_correction
    }
}

impl ScoreDrivenScale {
    /// Creates a scale initialized at one.
    pub fn new(config: ScoreDrivenScaleConfig) -> Self {
        Self {
            config,
            log_scale: 0.0,
        }
    }

    /// Restores a validated predictable log-scale state.
    pub fn from_log_scale(config: ScoreDrivenScaleConfig, log_scale: f64) -> HawkesResult<Self> {
        if !log_scale.is_finite() || log_scale.abs() > config.max_abs_log_scale {
            return Err(HawkesError::FittingError(
                "score-driven log scale must be finite and within its configured bound".to_string(),
            ));
        }
        Ok(Self { config, log_scale })
    }

    /// Configuration governing this state.
    pub fn config(&self) -> ScoreDrivenScaleConfig {
        self.config
    }

    /// Current multiplicative intensity scale, known before the next event.
    pub fn current_scale(&self) -> f64 {
        self.log_scale.exp()
    }

    /// Current log intensity scale.
    pub fn log_scale(&self) -> f64 {
        self.log_scale
    }

    /// Observes one unscaled compensator increment and updates future state.
    pub fn observe(&mut self, base_increment: f64) -> HawkesResult<ScoreDrivenObservation> {
        if !base_increment.is_finite() || base_increment < 0.0 {
            return Err(HawkesError::InvalidCompensator);
        }
        let scale = self.current_scale();
        let adjusted_increment = scale * base_increment;
        let correction = self.log_scale - (scale - 1.0) * base_increment;
        if !adjusted_increment.is_finite() || !correction.is_finite() {
            return Err(HawkesError::InvalidCompensator);
        }
        self.log_scale = (self.config.persistence * self.log_scale
            + self.config.score_step * (1.0 - adjusted_increment))
            .clamp(
                -self.config.max_abs_log_scale,
                self.config.max_abs_log_scale,
            );
        Ok(ScoreDrivenObservation {
            adjusted_increment,
            log_likelihood_correction: correction,
        })
    }
}

impl<'de> Deserialize<'de> for ScoreDrivenScale {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            config: ScoreDrivenScaleConfig,
            log_scale: f64,
        }
        let helper = Helper::deserialize(deserializer)?;
        Self::from_log_scale(helper.config, helper.log_scale).map_err(serde::de::Error::custom)
    }
}

impl ScoreDrivenObservation {
    /// Time-rescaled increment after applying the pre-event scale.
    pub fn adjusted_increment(&self) -> f64 {
        self.adjusted_increment
    }

    /// Exact conditional log-likelihood change relative to the base intensity.
    pub fn log_likelihood_correction(&self) -> f64 {
        self.log_likelihood_correction
    }
}

impl TimeRescalingCriteria {
    /// Creates explicit residual acceptance criteria.
    pub fn new(
        mean_tolerance: f64,
        max_ks_statistic: f64,
        max_abs_lag1_autocorrelation: f64,
        min_event_count: usize,
    ) -> HawkesResult<Self> {
        if !mean_tolerance.is_finite() || mean_tolerance < 0.0 {
            return Err(HawkesError::FittingError(
                "mean_tolerance must be finite and non-negative".to_string(),
            ));
        }
        if !max_ks_statistic.is_finite() || !(0.0..=1.0).contains(&max_ks_statistic) {
            return Err(HawkesError::FittingError(
                "max_ks_statistic must be finite and in [0, 1]".to_string(),
            ));
        }
        if !max_abs_lag1_autocorrelation.is_finite()
            || !(0.0..=1.0).contains(&max_abs_lag1_autocorrelation)
        {
            return Err(HawkesError::FittingError(
                "max_abs_lag1_autocorrelation must be finite and in [0, 1]".to_string(),
            ));
        }
        if min_event_count == 0 {
            return Err(HawkesError::FittingError(
                "min_event_count must be positive".to_string(),
            ));
        }
        Ok(Self {
            mean_tolerance,
            max_ks_statistic,
            max_abs_lag1_autocorrelation,
            min_event_count,
        })
    }

    /// Returns whether a report meets every configured effect-size gate.
    pub fn accepts(&self, report: &TimeRescalingReport) -> bool {
        report.event_count >= self.min_event_count
            && (report.mean_compensator - 1.0).abs() <= self.mean_tolerance
            && report.ks_statistic <= self.max_ks_statistic
            && report.lag1_autocorrelation.abs() <= self.max_abs_lag1_autocorrelation
    }

    /// Allowed absolute deviation of the mean compensator from one.
    pub fn mean_tolerance(&self) -> f64 {
        self.mean_tolerance
    }

    /// Maximum accepted uniform KS effect size.
    pub fn max_ks_statistic(&self) -> f64 {
        self.max_ks_statistic
    }

    /// Maximum accepted absolute lag-one residual autocorrelation.
    pub fn max_abs_lag1_autocorrelation(&self) -> f64 {
        self.max_abs_lag1_autocorrelation
    }

    /// Minimum residual sample size.
    pub fn min_event_count(&self) -> usize {
        self.min_event_count
    }
}

impl TimeRescalingReport {
    /// Number of scored event intervals.
    pub fn event_count(&self) -> usize {
        self.event_count
    }

    /// Mean compensator increment; a calibrated model has target one.
    pub fn mean_compensator(&self) -> f64 {
        self.mean_compensator
    }

    /// One-sample KS statistic after transforming increments to uniforms.
    pub fn ks_statistic(&self) -> f64 {
        self.ks_statistic
    }

    /// Approximate two-sided 5% KS critical value, `1.36 / sqrt(n)`.
    pub fn ks_critical_value_5pct(&self) -> f64 {
        self.ks_critical_value_5pct
    }

    /// Lag-one sample autocorrelation of compensator increments.
    pub fn lag1_autocorrelation(&self) -> f64 {
        self.lag1_autocorrelation
    }

    /// Returns whether the uniform KS statistic is below its approximate 5% threshold.
    pub fn passes_ks_5pct(&self) -> bool {
        self.ks_statistic <= self.ks_critical_value_5pct
    }
}

/// Conditional event-time score and its calibration diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalScore {
    log_likelihood: f64,
    compensator_increments: Vec<f64>,
    time_rescaling: TimeRescalingReport,
}

/// Conditional score and component-wise residual reports for a multivariate model.
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateConditionalScore {
    log_likelihood: f64,
    event_count: usize,
    component_increments: Vec<Vec<f64>>,
    component_reports: Vec<Option<TimeRescalingReport>>,
    pooled_report: Option<TimeRescalingReport>,
}

impl MultivariateConditionalScore {
    /// Conditional multivariate event-time log-likelihood.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Number of scored events across all components.
    pub fn event_count(&self) -> usize {
        self.event_count
    }

    /// Per-component compensator increments, excluding the first scored event of each component.
    pub fn component_increments(&self) -> &[Vec<f64>] {
        &self.component_increments
    }

    /// Per-component time-rescaling reports. Components with fewer than two scored events are `None`.
    pub fn component_reports(&self) -> &[Option<TimeRescalingReport>] {
        &self.component_reports
    }

    /// Pooled diagnostic across component residual sequences, if any are available.
    pub fn pooled_report(&self) -> Option<&TimeRescalingReport> {
        self.pooled_report.as_ref()
    }
}

impl ConditionalScore {
    /// Conditional event-time log-likelihood on the scored intervals.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Number of scored intervals.
    pub fn event_count(&self) -> usize {
        self.compensator_increments.len()
    }

    /// Compensator increments used by the time-rescaling diagnostic.
    pub fn compensator_increments(&self) -> &[f64] {
        &self.compensator_increments
    }

    /// Time-rescaling report for the scored intervals.
    pub fn time_rescaling(&self) -> &TimeRescalingReport {
        &self.time_rescaling
    }
}

/// Scores a fitted sum-exponential Hawkes model conditionally on earlier events.
///
/// `evaluation_start` is the index of the first event to score. Events before
/// it initialize the excitation recursion without contributing to the reported
/// likelihood. At least one earlier event is required as conditioning history.
pub fn score_hawkes(
    model: &SumExpHawkes,
    timestamps: &[f64],
    marks: &[f64],
    evaluation_start: usize,
) -> HawkesResult<ConditionalScore> {
    validate_score_inputs(timestamps, marks, evaluation_start)?;

    let mut recursions = vec![marks[0]; model.betas().len()];
    let mut log_likelihood = 0.0;
    let mut increments = Vec::with_capacity(timestamps.len() - evaluation_start);

    for (event_index, window) in timestamps.windows(2).enumerate() {
        let current_index = event_index + 1;
        let delta = window[1] - window[0];
        let mut integrated_intensity = model.integrated_baseline(window[0], window[1])?;
        let mut intensity = model.baseline_intensity(window[1])?;

        for ((recursion, &alpha), &beta) in
            recursions.iter_mut().zip(model.alphas()).zip(model.betas())
        {
            let decay = (-beta * delta).exp();
            integrated_intensity += alpha * *recursion * (1.0 - decay) / beta;
            *recursion *= decay;
            intensity += alpha * *recursion;
        }

        validate_intensity(intensity, integrated_intensity)?;
        if current_index >= evaluation_start {
            log_likelihood += intensity.ln() - integrated_intensity;
            increments.push(integrated_intensity);
        }

        for recursion in &mut recursions {
            *recursion += marks[current_index];
        }
    }

    build_score(log_likelihood, increments)
}

/// Scores a constant or periodic Poisson baseline on conditional event intervals.
pub fn score_poisson(
    baseline: &Baseline,
    timestamps: &[f64],
    evaluation_start: usize,
) -> HawkesResult<ConditionalScore> {
    validate_timestamps_and_start(timestamps, evaluation_start)?;

    let mut log_likelihood = 0.0;
    let mut increments = Vec::with_capacity(timestamps.len() - evaluation_start);
    for (event_index, window) in timestamps.windows(2).enumerate() {
        let current_index = event_index + 1;
        let intensity = baseline.intensity(window[1])?;
        let integrated_intensity = baseline.integrated_intensity(window[0], window[1])?;
        validate_intensity(intensity, integrated_intensity)?;

        if current_index >= evaluation_start {
            log_likelihood += intensity.ln() - integrated_intensity;
            increments.push(integrated_intensity);
        }
    }

    build_score(log_likelihood, increments)
}

/// Scores a multivariate Hawkes model after a completed warm-up timestamp batch.
///
/// Events at `evaluation_start_seconds` are warm-up history; events strictly
/// after it are scored. The boundary must match an event timestamp and may not
/// be the first or last timestamp.
pub fn score_multivariate_hawkes(
    model: &MultivariateSumExpHawkes,
    events: &[MultivariateEvent],
    evaluation_start_seconds: f64,
) -> HawkesResult<MultivariateConditionalScore> {
    validate_events(events, model.component_count())?;
    if !evaluation_start_seconds.is_finite() {
        return Err(HawkesError::NonFiniteTimestamp(evaluation_start_seconds));
    }
    let first_timestamp = events[0].timestamp_seconds();
    let last_timestamp = events[events.len() - 1].timestamp_seconds();
    if evaluation_start_seconds <= first_timestamp || evaluation_start_seconds >= last_timestamp {
        return Err(HawkesError::InvalidEvaluationStart {
            index: 0,
            event_count: events.len(),
        });
    }
    if !events
        .iter()
        .any(|event| event.timestamp_seconds() == evaluation_start_seconds)
    {
        return Err(HawkesError::EvaluationSplitsBatch);
    }

    let component_count = model.component_count();
    let kernel_count = model.kernel_count();
    let history_width = component_count * kernel_count;
    let mut recursion = vec![0.0; history_width];
    let mut accumulated = vec![0.0; component_count];
    let mut seen_scored_event = vec![false; component_count];
    let mut component_increments = vec![Vec::new(); component_count];
    let mut log_likelihood = 0.0;
    let mut event_count = 0;
    let mut batch_start = 0;
    let mut previous_timestamp = first_timestamp;

    while batch_start < events.len() {
        let timestamp = events[batch_start].timestamp_seconds();
        let mut scored_interval_total = 0.0;
        if batch_start > 0 {
            let delta = timestamp - previous_timestamp;
            if previous_timestamp >= evaluation_start_seconds {
                for (target, accumulator) in accumulated.iter_mut().enumerate() {
                    let mut integrated =
                        model.integrated_baseline(target, previous_timestamp, timestamp)?;
                    for source in 0..component_count {
                        for (kernel, &beta) in model.betas().iter().enumerate() {
                            let history = recursion[source * kernel_count + kernel];
                            let decay = (-beta * delta).exp();
                            integrated += model.alphas()[alpha_index(
                                target,
                                source,
                                kernel,
                                component_count,
                                kernel_count,
                            )] * history
                                * (1.0 - decay)
                                / beta;
                        }
                    }
                    *accumulator += integrated;
                    scored_interval_total += integrated;
                }
            }
            for source in 0..component_count {
                for (kernel, &beta) in model.betas().iter().enumerate() {
                    recursion[source * kernel_count + kernel] *= (-beta * delta).exp();
                }
            }
        }

        let batch_end = events[batch_start..]
            .iter()
            .position(|event| event.timestamp_seconds() != timestamp)
            .map_or(events.len(), |offset| batch_start + offset);
        if timestamp > evaluation_start_seconds {
            for event in &events[batch_start..batch_end] {
                let target = event.component();
                let mut intensity = model.baseline_intensity(target, timestamp)?;
                for source in 0..component_count {
                    for kernel in 0..kernel_count {
                        intensity += model.alphas()
                            [alpha_index(target, source, kernel, component_count, kernel_count)]
                            * recursion[source * kernel_count + kernel];
                    }
                }
                if !intensity.is_finite() || intensity <= 0.0 {
                    return Err(HawkesError::InvalidConditionalIntensity(intensity));
                }
                log_likelihood += intensity.ln();
                event_count += 1;
                if seen_scored_event[target] {
                    component_increments[target].push(accumulated[target]);
                } else {
                    seen_scored_event[target] = true;
                }
                accumulated[target] = 0.0;
            }
            log_likelihood -= scored_interval_total;
        }

        for event in &events[batch_start..batch_end] {
            let offset = event.component() * kernel_count;
            for value in &mut recursion[offset..offset + kernel_count] {
                *value += event.mark();
            }
        }
        previous_timestamp = timestamp;
        batch_start = batch_end;
    }

    if !log_likelihood.is_finite() || event_count == 0 {
        return Err(HawkesError::EmptyDiagnostics);
    }
    let component_reports = component_increments
        .iter()
        .map(|increments| {
            if increments.is_empty() {
                Ok(None)
            } else {
                diagnose_compensators(increments).map(Some)
            }
        })
        .collect::<HawkesResult<Vec<_>>>()?;
    let pooled = component_increments
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let pooled_report = if pooled.is_empty() {
        None
    } else {
        Some(diagnose_compensators(&pooled)?)
    };
    Ok(MultivariateConditionalScore {
        log_likelihood,
        event_count,
        component_increments,
        component_reports,
        pooled_report,
    })
}

/// Computes time-rescaling diagnostics from positive compensator increments.
pub fn diagnose_compensators(increments: &[f64]) -> HawkesResult<TimeRescalingReport> {
    if increments.is_empty() {
        return Err(HawkesError::EmptyDiagnostics);
    }
    if increments
        .iter()
        .any(|&increment| !increment.is_finite() || increment < 0.0)
    {
        return Err(HawkesError::InvalidCompensator);
    }

    let event_count = increments.len();
    let sample_size = event_count as f64;
    let mean_compensator = increments.iter().sum::<f64>() / sample_size;
    let mut transformed = increments
        .iter()
        .map(|&increment| 1.0 - (-increment).exp())
        .collect::<Vec<_>>();
    transformed.sort_by(f64::total_cmp);
    let ks_statistic = transformed
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let lower = index as f64 / sample_size;
            let upper = (index + 1) as f64 / sample_size;
            (value - lower).abs().max((upper - value).abs())
        })
        .fold(0.0, f64::max);

    Ok(TimeRescalingReport {
        event_count,
        mean_compensator,
        ks_statistic,
        ks_critical_value_5pct: 1.36 / sample_size.sqrt(),
        lag1_autocorrelation: lag1_autocorrelation(increments, mean_compensator),
    })
}

fn validate_score_inputs(
    timestamps: &[f64],
    marks: &[f64],
    evaluation_start: usize,
) -> HawkesResult<()> {
    if timestamps.len() != marks.len() {
        return Err(HawkesError::VolumeDimensionMismatch(
            marks.len(),
            timestamps.len(),
        ));
    }
    validate_timestamps_and_start(timestamps, evaluation_start)?;
    for &mark in marks {
        if !mark.is_finite() {
            return Err(HawkesError::NonFiniteParameter("volume", mark));
        }
        if mark < 0.0 {
            return Err(HawkesError::InvalidVolume(mark));
        }
    }
    Ok(())
}

fn validate_timestamps_and_start(timestamps: &[f64], evaluation_start: usize) -> HawkesResult<()> {
    if timestamps.len() < 2 || evaluation_start == 0 || evaluation_start >= timestamps.len() {
        return Err(HawkesError::InvalidEvaluationStart {
            index: evaluation_start,
            event_count: timestamps.len(),
        });
    }
    if let Some(&timestamp) = timestamps.iter().find(|timestamp| !timestamp.is_finite()) {
        return Err(HawkesError::NonFiniteTimestamp(timestamp));
    }
    if timestamps.windows(2).any(|window| window[1] < window[0]) {
        return Err(HawkesError::UnsortedTimestamps);
    }
    Ok(())
}

fn validate_intensity(intensity: f64, integrated_intensity: f64) -> HawkesResult<()> {
    if !intensity.is_finite() || intensity <= 0.0 {
        return Err(HawkesError::InvalidConditionalIntensity(intensity));
    }
    if !integrated_intensity.is_finite() || integrated_intensity < 0.0 {
        return Err(HawkesError::InvalidCompensator);
    }
    Ok(())
}

fn build_score(log_likelihood: f64, increments: Vec<f64>) -> HawkesResult<ConditionalScore> {
    if !log_likelihood.is_finite() {
        return Err(HawkesError::FittingError(
            "conditional log-likelihood is non-finite".to_string(),
        ));
    }
    let time_rescaling = diagnose_compensators(&increments)?;
    Ok(ConditionalScore {
        log_likelihood,
        compensator_increments: increments,
        time_rescaling,
    })
}

fn lag1_autocorrelation(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let denominator = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    if denominator == 0.0 {
        return 0.0;
    }
    values
        .windows(2)
        .map(|window| (window[0] - mean) * (window[1] - mean))
        .sum::<f64>()
        / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MultivariateEvent, MultivariateSumExpHawkes, PeriodicShape};

    #[test]
    fn constant_poisson_score_matches_closed_form() {
        let timestamps = vec![0.0, 1.0, 2.0, 3.0];
        let baseline = Baseline::constant(2.0).unwrap();
        let score = score_poisson(&baseline, &timestamps, 1).unwrap();

        assert!((score.log_likelihood() - 3.0 * (2.0_f64.ln() - 2.0)).abs() < 1.0e-12);
        assert_eq!(score.compensator_increments(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn periodic_poisson_integrates_each_interval() {
        let shape = PeriodicShape::new(2.0, vec![0.5, 1.5]).unwrap();
        let baseline = Baseline::periodic(2.0, shape).unwrap();
        let score = score_poisson(&baseline, &[0.0, 1.0, 2.0], 1).unwrap();

        assert_eq!(score.compensator_increments(), &[1.0, 3.0]);
        assert!((score.log_likelihood() - (3.0_f64.ln() - 4.0)).abs() < 1.0e-12);
    }

    #[test]
    fn hawkes_score_uses_warmup_events() {
        let model = SumExpHawkes::new(1.0, vec![0.2], vec![1.0]).unwrap();
        let score = score_hawkes(&model, &[0.0, 1.0, 2.0], &[1.0; 3], 2).unwrap();
        let decay = (-1.0_f64).exp();
        let recursion = decay * (decay + 1.0);
        let intensity = 1.0 + 0.2 * recursion;
        let integral = 1.0 + 0.2 * (decay + 1.0) * (1.0 - decay);

        assert!((score.log_likelihood() - (intensity.ln() - integral)).abs() < 1.0e-12);
    }

    #[test]
    fn multivariate_score_integrates_all_components_once() {
        let model =
            MultivariateSumExpHawkes::new(vec![1.0, 1.0], vec![0.0; 4], vec![1.0], vec![1.0, 1.0])
                .unwrap();
        let events = (0..4)
            .flat_map(|time| {
                (0..2).map(move |component| {
                    MultivariateEvent::new(time as f64, component, 1.0).unwrap()
                })
            })
            .collect::<Vec<_>>();
        let score = score_multivariate_hawkes(&model, &events, 1.0).unwrap();

        assert_eq!(score.event_count(), 4);
        assert!((score.log_likelihood() + 4.0).abs() < 1.0e-12);
        assert_eq!(score.component_increments()[0], vec![1.0]);
        assert_eq!(score.component_increments()[1], vec![1.0]);
    }

    #[test]
    fn explicit_residual_criteria_apply_all_gates() {
        let criteria = TimeRescalingCriteria::new(0.1, 0.1, 0.1, 3).unwrap();
        let passing = TimeRescalingReport {
            event_count: 3,
            mean_compensator: 1.0,
            ks_statistic: 0.05,
            ks_critical_value_5pct: 0.78,
            lag1_autocorrelation: 0.05,
        };
        let correlated = TimeRescalingReport {
            lag1_autocorrelation: 0.2,
            ..passing.clone()
        };

        assert!(criteria.accepts(&passing));
        assert!(!criteria.accepts(&correlated));
        assert!(TimeRescalingCriteria::new(-0.1, 0.1, 0.1, 3).is_err());
    }

    #[test]
    fn zero_step_score_driven_scale_is_identity() {
        let config = ScoreDrivenScaleConfig::new(0.9, 0.0, 2.0).unwrap();
        let mut scale = ScoreDrivenScale::new(config);

        for increment in [0.2, 1.0, 3.0] {
            let observation = scale.observe(increment).unwrap();
            assert_eq!(observation.adjusted_increment(), increment);
            assert_eq!(observation.log_likelihood_correction(), 0.0);
        }
        assert_eq!(scale.current_scale(), 1.0);
    }

    #[test]
    fn score_driven_scale_updates_only_future_events() {
        let config = ScoreDrivenScaleConfig::new(0.0, 0.5, 2.0).unwrap();
        let mut scale = ScoreDrivenScale::new(config);

        let first = scale.observe(0.5).unwrap();
        let expected_scale = 0.25_f64.exp();
        let second = scale.observe(1.0).unwrap();

        assert_eq!(first.adjusted_increment(), 0.5);
        assert!((second.adjusted_increment() - expected_scale).abs() < 1.0e-12);
        assert!(ScoreDrivenScaleConfig::new(1.0, 0.1, 2.0).is_err());
    }

    #[test]
    fn fitted_weibull_time_change_normalizes_mean_hazard() {
        let increments = [0.1, 0.3, 0.7, 1.2, 2.0, 4.0];
        let time_change = WeibullTimeChange::fit(&increments).unwrap();
        let adjusted = increments
            .iter()
            .map(|&increment| time_change.observe(increment).unwrap().adjusted_increment())
            .collect::<Vec<_>>();
        let mean = adjusted.iter().sum::<f64>() / adjusted.len() as f64;

        assert!((mean - 1.0).abs() < 1.0e-12);
        assert!(time_change.shape().is_finite());
        assert!(time_change.scale().is_finite());
        assert!(WeibullTimeChange::new(0.0, 1.0).is_err());
    }

    #[test]
    fn weibull_likelihood_correction_includes_compensator_change() {
        let time_change = WeibullTimeChange::new(1.5, 0.8).unwrap();
        let base_increment = 1.2;
        let observation = time_change.observe(base_increment).unwrap();
        let adjusted = (base_increment / 0.8).powf(1.5);
        let log_hazard = 1.5_f64.ln() - 1.5 * 0.8_f64.ln() + (1.5 - 1.0) * base_increment.ln();

        assert!((observation.adjusted_increment() - adjusted).abs() < 1.0e-12);
        assert!(
            (observation.log_likelihood_correction() - (log_hazard - adjusted + base_increment))
                .abs()
                < 1.0e-12
        );
    }
}
