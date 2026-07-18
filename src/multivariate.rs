//! Marked multivariate sum-exponential Hawkes processes.

use crate::{HawkesError, HawkesResult, PeriodicShape};
use serde::{Deserialize, Deserializer, Serialize};

/// Controls multivariate stationarity margin, baseline floor, and optimizer work.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultivariateFitConfig {
    max_source_offspring: f64,
    min_baseline_fraction: f64,
    max_iterations: u64,
}

impl MultivariateFitConfig {
    /// Creates a fitting configuration.
    ///
    /// `max_source_offspring` must be in `(0, 1)`. `min_baseline_fraction`
    /// is a fraction of each component's empirical rate and must be in `[0, 1)`.
    pub fn new(
        max_source_offspring: f64,
        min_baseline_fraction: f64,
        max_iterations: u64,
    ) -> HawkesResult<Self> {
        if !max_source_offspring.is_finite()
            || max_source_offspring <= 0.0
            || max_source_offspring >= 1.0
        {
            return Err(HawkesError::FittingError(
                "max_source_offspring must be finite and in (0, 1)".to_string(),
            ));
        }
        if !min_baseline_fraction.is_finite() || !(0.0..1.0).contains(&min_baseline_fraction) {
            return Err(HawkesError::FittingError(
                "min_baseline_fraction must be finite and in [0, 1)".to_string(),
            ));
        }
        if max_iterations == 0 {
            return Err(HawkesError::FittingError(
                "max_iterations must be positive".to_string(),
            ));
        }
        Ok(Self {
            max_source_offspring,
            min_baseline_fraction,
            max_iterations,
        })
    }

    /// Maximum expected offspring mass from any one source component.
    pub fn max_source_offspring(&self) -> f64 {
        self.max_source_offspring
    }

    /// Minimum baseline as a fraction of the component's empirical rate.
    pub fn min_baseline_fraction(&self) -> f64 {
        self.min_baseline_fraction
    }

    /// Maximum L-BFGS iterations.
    pub fn max_iterations(&self) -> u64 {
        self.max_iterations
    }
}

impl Default for MultivariateFitConfig {
    fn default() -> Self {
        Self {
            max_source_offspring: 0.95,
            min_baseline_fraction: 0.05,
            max_iterations: 150,
        }
    }
}

/// An offline marked event assigned to one process component.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct MultivariateEvent {
    timestamp_seconds: f64,
    component: usize,
    mark: f64,
}

impl<'de> Deserialize<'de> for MultivariateEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            timestamp_seconds: f64,
            component: usize,
            mark: f64,
        }

        let helper = Helper::deserialize(deserializer)?;
        Self::new(helper.timestamp_seconds, helper.component, helper.mark)
            .map_err(serde::de::Error::custom)
    }
}

impl MultivariateEvent {
    /// Creates an event. Component range is validated when fitting or scoring.
    pub fn new(timestamp_seconds: f64, component: usize, mark: f64) -> HawkesResult<Self> {
        if !timestamp_seconds.is_finite() {
            return Err(HawkesError::NonFiniteTimestamp(timestamp_seconds));
        }
        validate_mark(mark)?;
        Ok(Self {
            timestamp_seconds,
            component,
            mark,
        })
    }

    /// Event time in seconds.
    pub fn timestamp_seconds(&self) -> f64 {
        self.timestamp_seconds
    }

    /// Zero-based process component.
    pub fn component(&self) -> usize {
        self.component
    }

    /// Non-negative event mark.
    pub fn mark(&self) -> f64 {
        self.mark
    }
}

/// A component and mark used by an online timestamp batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComponentMark {
    component: usize,
    mark: f64,
}

impl ComponentMark {
    /// Creates an online component mark. Component range is checked on update.
    pub fn new(component: usize, mark: f64) -> HawkesResult<Self> {
        validate_mark(mark)?;
        Ok(Self { component, mark })
    }

    /// Zero-based process component.
    pub fn component(&self) -> usize {
        self.component
    }

    /// Non-negative event mark.
    pub fn mark(&self) -> f64 {
        self.mark
    }
}

/// A marked multivariate Hawkes process with shared exponential decay scales.
///
/// Flattened `alphas` use `(target, source, kernel)` order. Events in one
/// [`Self::update_batch`] share the same pre-event history and therefore do not
/// spuriously excite one another at zero lag.
#[derive(Debug, Clone, Serialize)]
pub struct MultivariateSumExpHawkes {
    mus: Vec<f64>,
    alphas: Vec<f64>,
    betas: Vec<f64>,
    expected_marks: Vec<f64>,
    seasonal_shapes: Vec<Option<PeriodicShape>>,
    excitations: Vec<f64>,
    #[serde(skip)]
    next_excitations: Vec<f64>,
    #[serde(skip)]
    current_intensities: Vec<f64>,
    #[serde(skip)]
    next_intensities: Vec<f64>,
    #[serde(skip)]
    seen_components: Vec<bool>,
    last_timestamp_us: Option<u64>,
}

impl MultivariateSumExpHawkes {
    /// Fits a marked model with one fixed unit-mean seasonal shape per component.
    ///
    /// The branching parameterization constrains every source column's expected
    /// offspring mass below one, which is sufficient for stationarity.
    pub fn fit_marked_seasonal(
        events: Vec<MultivariateEvent>,
        component_count: usize,
        fixed_betas: Vec<f64>,
        expected_marks: Vec<f64>,
        seasonal_shapes: Vec<PeriodicShape>,
    ) -> HawkesResult<Self> {
        Self::fit_marked_seasonal_with_config(
            events,
            component_count,
            fixed_betas,
            expected_marks,
            seasonal_shapes,
            MultivariateFitConfig::default(),
        )
    }

    /// Fits a seasonal marked model with explicit production constraints.
    pub fn fit_marked_seasonal_with_config(
        events: Vec<MultivariateEvent>,
        component_count: usize,
        fixed_betas: Vec<f64>,
        expected_marks: Vec<f64>,
        seasonal_shapes: Vec<PeriodicShape>,
        config: MultivariateFitConfig,
    ) -> HawkesResult<Self> {
        let (mus, alphas) = crate::multivariate_fitting::fit_marked_seasonal(
            events,
            component_count,
            fixed_betas.clone(),
            expected_marks.clone(),
            &seasonal_shapes,
            config,
        )?;
        Self::new_with_seasonal_shapes(mus, alphas, fixed_betas, expected_marks, seasonal_shapes)
    }

    /// Re-estimates baseline rates on a recent calibration window.
    ///
    /// Excitation parameters, decay rates, mark expectations, and seasonal
    /// shapes remain fixed, so this convex update cannot change stationarity.
    /// The returned model has reset online state. The first timestamp batch is
    /// conditioning history, consistent with the fitting likelihood.
    pub fn recalibrate_baselines(
        &self,
        events: &[MultivariateEvent],
        min_baseline_fraction: f64,
    ) -> HawkesResult<Self> {
        if !min_baseline_fraction.is_finite() || !(0.0..1.0).contains(&min_baseline_fraction) {
            return Err(HawkesError::FittingError(
                "min_baseline_fraction must be finite and in [0, 1)".to_string(),
            ));
        }
        let mus = recalibrated_mus(self, events, min_baseline_fraction)?;
        Self::new_validated(
            mus,
            self.alphas.clone(),
            self.betas.clone(),
            self.expected_marks.clone(),
            self.seasonal_shapes.clone(),
        )
    }

    /// Creates a constant-baseline multivariate model.
    pub fn new(
        mus: Vec<f64>,
        alphas: Vec<f64>,
        betas: Vec<f64>,
        expected_marks: Vec<f64>,
    ) -> HawkesResult<Self> {
        let component_count = mus.len();
        Self::new_validated(
            mus,
            alphas,
            betas,
            expected_marks,
            vec![None; component_count],
        )
    }

    /// Creates a model with one unit-mean periodic baseline shape per component.
    pub fn new_with_seasonal_shapes(
        mus: Vec<f64>,
        alphas: Vec<f64>,
        betas: Vec<f64>,
        expected_marks: Vec<f64>,
        seasonal_shapes: Vec<PeriodicShape>,
    ) -> HawkesResult<Self> {
        Self::new_validated(
            mus,
            alphas,
            betas,
            expected_marks,
            seasonal_shapes.into_iter().map(Some).collect(),
        )
    }

    fn new_validated(
        mus: Vec<f64>,
        alphas: Vec<f64>,
        betas: Vec<f64>,
        expected_marks: Vec<f64>,
        seasonal_shapes: Vec<Option<PeriodicShape>>,
    ) -> HawkesResult<Self> {
        let component_count = mus.len();
        validate_dimensions(
            component_count,
            &alphas,
            &betas,
            &expected_marks,
            &seasonal_shapes,
        )?;
        for &mu in &mus {
            validate_nonnegative("mu", mu)?;
        }
        for &alpha in &alphas {
            validate_nonnegative("alpha", alpha)?;
        }
        for &beta in &betas {
            if !beta.is_finite() {
                return Err(HawkesError::NonFiniteParameter("beta", beta));
            }
            if beta <= 0.0 {
                return Err(HawkesError::InvalidBeta(beta));
            }
        }
        for &expected_mark in &expected_marks {
            if !expected_mark.is_finite() || expected_mark <= 0.0 {
                return Err(HawkesError::InvalidFittingExpectedVolume(expected_mark));
            }
        }

        let branching = branching_matrix(component_count, &alphas, &betas, &expected_marks);
        let spectral_bound = spectral_radius_upper_bound(&branching, component_count)?;
        if spectral_bound >= 1.0 {
            return Err(HawkesError::InvalidBranchingRatio(spectral_bound));
        }

        let state_len = alphas.len();
        let initial_intensities = mus.clone();
        Ok(Self {
            mus,
            alphas,
            betas,
            expected_marks,
            seasonal_shapes,
            excitations: vec![0.0; state_len],
            next_excitations: vec![0.0; state_len],
            current_intensities: initial_intensities,
            next_intensities: vec![0.0; component_count],
            seen_components: vec![false; component_count],
            last_timestamp_us: None,
        })
    }

    /// Applies simultaneous events and returns intensities immediately afterward.
    ///
    /// A component may occur at most once in a batch; aggregate duplicate
    /// component records before calling this method. Each call must be strictly
    /// later than the previous call; include all simultaneous events in one batch.
    pub fn update_batch(
        &mut self,
        timestamp_us: u64,
        events: &[ComponentMark],
    ) -> HawkesResult<&[f64]> {
        self.seen_components.fill(false);
        for event in events {
            validate_component(event.component, self.component_count())?;
            validate_mark(event.mark)?;
            if self.seen_components[event.component] {
                return Err(HawkesError::DuplicateComponentAtTimestamp {
                    component: event.component,
                    timestamp: timestamp_us as f64 / 1_000_000.0,
                });
            }
            self.seen_components[event.component] = true;
        }

        if let Some(previous) = self.last_timestamp_us {
            if timestamp_us == previous {
                return Err(HawkesError::DuplicateTimestamp(
                    timestamp_us as f64 / 1_000_000.0,
                ));
            }
            if timestamp_us < previous {
                return Err(HawkesError::NonMonotonicTimestamp {
                    previous_us: previous,
                    current_us: timestamp_us,
                });
            }
            let delta = (timestamp_us - previous) as f64 / 1_000_000.0;
            let kernel_count = self.kernel_count();
            for (index, candidate) in self.next_excitations.iter_mut().enumerate() {
                let kernel = index % kernel_count;
                *candidate = self.excitations[index] * (-self.betas[kernel] * delta).exp();
            }
        } else {
            self.next_excitations.fill(0.0);
        }

        let component_count = self.component_count();
        let kernel_count = self.kernel_count();
        for event in events {
            for target in 0..component_count {
                for kernel in 0..kernel_count {
                    let index = alpha_index(
                        target,
                        event.component,
                        kernel,
                        component_count,
                        kernel_count,
                    );
                    self.next_excitations[index] += self.alphas[index] * event.mark;
                }
            }
        }
        if self
            .next_excitations
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(HawkesError::InvalidExcitation(f64::INFINITY));
        }

        let timestamp_seconds = timestamp_us as f64 / 1_000_000.0;
        for target in 0..component_count {
            let excitation = (0..component_count)
                .flat_map(|source| {
                    (0..kernel_count).map(move |kernel| {
                        alpha_index(target, source, kernel, component_count, kernel_count)
                    })
                })
                .map(|index| self.next_excitations[index])
                .sum::<f64>();
            let intensity = self.baseline_intensity(target, timestamp_seconds)? + excitation;
            if !intensity.is_finite() || intensity < 0.0 {
                return Err(HawkesError::InvalidConditionalIntensity(intensity));
            }
            self.next_intensities[target] = intensity;
        }

        self.excitations.copy_from_slice(&self.next_excitations);
        self.current_intensities
            .copy_from_slice(&self.next_intensities);
        self.last_timestamp_us = Some(timestamp_us);
        Ok(&self.current_intensities)
    }

    /// Number of modeled event components.
    pub fn component_count(&self) -> usize {
        self.mus.len()
    }

    /// Number of shared exponential decay scales.
    pub fn kernel_count(&self) -> usize {
        self.betas.len()
    }

    /// Mean baseline rates by component.
    pub fn mus(&self) -> &[f64] {
        &self.mus
    }

    /// Flattened `(target, source, kernel)` excitation coefficients.
    pub fn alphas(&self) -> &[f64] {
        &self.alphas
    }

    /// Shared exponential decay rates.
    pub fn betas(&self) -> &[f64] {
        &self.betas
    }

    /// Expected marks by source component.
    pub fn expected_marks(&self) -> &[f64] {
        &self.expected_marks
    }

    /// Seasonal shapes by component; `None` means a constant baseline.
    pub fn seasonal_shapes(&self) -> &[Option<PeriodicShape>] {
        &self.seasonal_shapes
    }

    /// Returns one excitation coefficient.
    pub fn alpha(&self, target: usize, source: usize, kernel: usize) -> HawkesResult<f64> {
        validate_component(target, self.component_count())?;
        validate_component(source, self.component_count())?;
        if kernel >= self.kernel_count() {
            return Err(HawkesError::DimensionMismatch(
                kernel + 1,
                self.kernel_count(),
            ));
        }
        Ok(self.alphas[alpha_index(
            target,
            source,
            kernel,
            self.component_count(),
            self.kernel_count(),
        )])
    }

    /// Effective branching matrix in row-target, column-source order.
    pub fn branching_matrix(&self) -> Vec<f64> {
        branching_matrix(
            self.component_count(),
            &self.alphas,
            &self.betas,
            &self.expected_marks,
        )
    }

    /// Conservative upper bound on the branching matrix spectral radius.
    pub fn branching_spectral_radius_upper_bound(&self) -> f64 {
        spectral_radius_upper_bound(&self.branching_matrix(), self.component_count())
            .unwrap_or(f64::INFINITY)
    }

    /// Current post-event intensities by component.
    pub fn current_intensities(&self) -> &[f64] {
        &self.current_intensities
    }

    /// Most recent online timestamp.
    pub fn last_timestamp_us(&self) -> Option<u64> {
        self.last_timestamp_us
    }

    pub(crate) fn baseline_intensity(
        &self,
        component: usize,
        timestamp_seconds: f64,
    ) -> HawkesResult<f64> {
        validate_component(component, self.component_count())?;
        if !timestamp_seconds.is_finite() {
            return Err(HawkesError::NonFiniteTimestamp(timestamp_seconds));
        }
        match &self.seasonal_shapes[component] {
            Some(shape) => Ok(self.mus[component] * shape.multiplier(timestamp_seconds)?),
            None => Ok(self.mus[component]),
        }
    }

    pub(crate) fn integrated_baseline(
        &self,
        component: usize,
        start: f64,
        end: f64,
    ) -> HawkesResult<f64> {
        validate_component(component, self.component_count())?;
        match &self.seasonal_shapes[component] {
            Some(shape) => Ok(self.mus[component] * shape.integrated_multiplier(start, end)?),
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
                Ok(self.mus[component] * (end - start))
            }
        }
    }
}

fn recalibrated_mus(
    model: &MultivariateSumExpHawkes,
    events: &[MultivariateEvent],
    min_baseline_fraction: f64,
) -> HawkesResult<Vec<f64>> {
    validate_events(events, model.component_count())?;
    let component_count = model.component_count();
    let kernel_count = model.kernel_count();
    let start = events[0].timestamp_seconds;
    let end = events[events.len() - 1].timestamp_seconds;
    let duration = end - start;
    let mut terms: Vec<Vec<(f64, f64)>> = vec![Vec::new(); component_count];
    let mut recursion = vec![0.0; component_count * kernel_count];
    let mut batch_start = 0;
    let mut previous_timestamp = start;

    while batch_start < events.len() {
        let timestamp = events[batch_start].timestamp_seconds;
        if batch_start > 0 {
            let delta = timestamp - previous_timestamp;
            for source in 0..component_count {
                for (kernel, &beta) in model.betas.iter().enumerate() {
                    recursion[source * kernel_count + kernel] *= (-beta * delta).exp();
                }
            }
        }
        let batch_end = events[batch_start..]
            .iter()
            .position(|event| event.timestamp_seconds != timestamp)
            .map_or(events.len(), |offset| batch_start + offset);
        if timestamp > start {
            for event in &events[batch_start..batch_end] {
                let target = event.component;
                let baseline_multiplier = match &model.seasonal_shapes[target] {
                    Some(shape) => shape.multiplier(timestamp)?,
                    None => 1.0,
                };
                let excitation = (0..component_count)
                    .flat_map(|source| (0..kernel_count).map(move |kernel| (source, kernel)))
                    .map(|(source, kernel)| {
                        model.alphas
                            [alpha_index(target, source, kernel, component_count, kernel_count)]
                            * recursion[source * kernel_count + kernel]
                    })
                    .sum::<f64>();
                if !excitation.is_finite() || excitation < 0.0 {
                    return Err(HawkesError::InvalidExcitation(excitation));
                }
                terms[target].push((baseline_multiplier, excitation));
            }
        }
        for event in &events[batch_start..batch_end] {
            let offset = event.component * kernel_count;
            for value in &mut recursion[offset..offset + kernel_count] {
                *value += event.mark;
            }
        }
        previous_timestamp = timestamp;
        batch_start = batch_end;
    }

    terms
        .iter()
        .enumerate()
        .map(|(component, component_terms)| {
            if component_terms.is_empty() {
                return Err(HawkesError::FittingError(format!(
                    "component {component} has no events after the conditioning batch"
                )));
            }
            let baseline_integral = match &model.seasonal_shapes[component] {
                Some(shape) => shape.integrated_multiplier(start, end)?,
                None => duration,
            };
            let empirical_rate = component_terms.len() as f64 / duration;
            let lower = (min_baseline_fraction * empirical_rate).max(empirical_rate * 1.0e-12);
            let derivative = |mu: f64| {
                baseline_integral
                    - component_terms
                        .iter()
                        .map(|&(multiplier, excitation)| {
                            multiplier / (mu * multiplier + excitation)
                        })
                        .sum::<f64>()
            };
            if derivative(lower) >= 0.0 {
                return Ok(lower);
            }
            let mut upper = empirical_rate.max(lower * 2.0);
            for _ in 0..128 {
                if derivative(upper) >= 0.0 {
                    let mut lower_bound = lower;
                    let mut upper_bound = upper;
                    for _ in 0..80 {
                        let midpoint = 0.5 * (lower_bound + upper_bound);
                        if derivative(midpoint) < 0.0 {
                            lower_bound = midpoint;
                        } else {
                            upper_bound = midpoint;
                        }
                    }
                    return Ok(0.5 * (lower_bound + upper_bound));
                }
                upper *= 2.0;
                if !upper.is_finite() {
                    break;
                }
            }
            Err(HawkesError::FittingError(format!(
                "baseline calibration failed for component {component}"
            )))
        })
        .collect()
}

impl<'de> Deserialize<'de> for MultivariateSumExpHawkes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            mus: Vec<f64>,
            alphas: Vec<f64>,
            betas: Vec<f64>,
            expected_marks: Vec<f64>,
            seasonal_shapes: Vec<Option<PeriodicShape>>,
            excitations: Vec<f64>,
            last_timestamp_us: Option<u64>,
        }

        let helper = Helper::deserialize(deserializer)?;
        let mut model = Self::new_validated(
            helper.mus,
            helper.alphas,
            helper.betas,
            helper.expected_marks,
            helper.seasonal_shapes,
        )
        .map_err(serde::de::Error::custom)?;
        if helper.excitations.len() != model.excitations.len()
            || helper
                .excitations
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(serde::de::Error::custom(
                "invalid multivariate excitation state",
            ));
        }
        model.excitations = helper.excitations;
        model.last_timestamp_us = helper.last_timestamp_us;
        if let Some(timestamp) = model.last_timestamp_us {
            let timestamp_seconds = timestamp as f64 / 1_000_000.0;
            for target in 0..model.component_count() {
                let mut intensity = model
                    .baseline_intensity(target, timestamp_seconds)
                    .map_err(serde::de::Error::custom)?;
                for source in 0..model.component_count() {
                    for kernel in 0..model.kernel_count() {
                        intensity += model.excitations[alpha_index(
                            target,
                            source,
                            kernel,
                            model.component_count(),
                            model.kernel_count(),
                        )];
                    }
                }
                if !intensity.is_finite() {
                    return Err(serde::de::Error::custom(
                        "invalid multivariate intensity state",
                    ));
                }
                model.current_intensities[target] = intensity;
            }
        }
        Ok(model)
    }
}

pub(crate) fn alpha_index(
    target: usize,
    source: usize,
    kernel: usize,
    component_count: usize,
    kernel_count: usize,
) -> usize {
    (target * component_count + source) * kernel_count + kernel
}

pub(crate) fn validate_events(
    events: &[MultivariateEvent],
    component_count: usize,
) -> HawkesResult<()> {
    if component_count == 0 {
        return Err(HawkesError::InvalidComponentCount(component_count));
    }
    if events.is_empty() {
        return Err(HawkesError::EmptyTimestamps);
    }
    for event in events {
        if !event.timestamp_seconds.is_finite() {
            return Err(HawkesError::NonFiniteTimestamp(event.timestamp_seconds));
        }
        validate_component(event.component, component_count)?;
        validate_mark(event.mark)?;
    }
    if events
        .windows(2)
        .any(|window| window[1].timestamp_seconds < window[0].timestamp_seconds)
    {
        return Err(HawkesError::UnsortedTimestamps);
    }
    if events[events.len() - 1].timestamp_seconds <= events[0].timestamp_seconds {
        return Err(HawkesError::InvalidObservationWindow);
    }
    let mut batch_start = 0;
    let mut seen_components = vec![false; component_count];
    while batch_start < events.len() {
        let timestamp = events[batch_start].timestamp_seconds;
        let batch_end = events[batch_start..]
            .iter()
            .position(|event| event.timestamp_seconds != timestamp)
            .map_or(events.len(), |offset| batch_start + offset);
        seen_components.fill(false);
        for event in &events[batch_start..batch_end] {
            if seen_components[event.component] {
                return Err(HawkesError::DuplicateComponentAtTimestamp {
                    component: event.component,
                    timestamp,
                });
            }
            seen_components[event.component] = true;
        }
        batch_start = batch_end;
    }
    Ok(())
}

fn validate_dimensions(
    component_count: usize,
    alphas: &[f64],
    betas: &[f64],
    expected_marks: &[f64],
    seasonal_shapes: &[Option<PeriodicShape>],
) -> HawkesResult<()> {
    if component_count == 0 {
        return Err(HawkesError::InvalidComponentCount(component_count));
    }
    if betas.is_empty() {
        return Err(HawkesError::FittingError(
            "fixed_betas must contain at least one decay rate".to_string(),
        ));
    }
    for (name, actual) in [
        ("expected_marks", expected_marks.len()),
        ("seasonal_shapes", seasonal_shapes.len()),
    ] {
        if actual != component_count {
            return Err(HawkesError::ComponentDimensionMismatch {
                name,
                actual,
                expected: component_count,
            });
        }
    }
    let expected = component_count
        .checked_mul(component_count)
        .and_then(|value| value.checked_mul(betas.len()))
        .ok_or_else(|| HawkesError::FittingError("alpha tensor size overflow".to_string()))?;
    if alphas.len() != expected {
        return Err(HawkesError::InvalidAlphaTensor {
            actual: alphas.len(),
            expected,
        });
    }
    Ok(())
}

fn validate_component(component: usize, component_count: usize) -> HawkesResult<()> {
    if component >= component_count {
        return Err(HawkesError::InvalidComponent {
            component,
            component_count,
        });
    }
    Ok(())
}

fn validate_mark(mark: f64) -> HawkesResult<()> {
    if !mark.is_finite() {
        return Err(HawkesError::NonFiniteParameter("mark", mark));
    }
    if mark < 0.0 {
        return Err(HawkesError::InvalidVolume(mark));
    }
    Ok(())
}

fn validate_nonnegative(name: &'static str, value: f64) -> HawkesResult<()> {
    if !value.is_finite() {
        return Err(HawkesError::NonFiniteParameter(name, value));
    }
    if value < 0.0 {
        return Err(match name {
            "mu" => HawkesError::InvalidMu(value),
            "alpha" => HawkesError::InvalidAlpha(value),
            _ => HawkesError::NonFiniteParameter(name, value),
        });
    }
    Ok(())
}

fn branching_matrix(
    component_count: usize,
    alphas: &[f64],
    betas: &[f64],
    expected_marks: &[f64],
) -> Vec<f64> {
    let mut matrix = vec![0.0; component_count * component_count];
    for target in 0..component_count {
        for source in 0..component_count {
            matrix[target * component_count + source] = betas
                .iter()
                .enumerate()
                .map(|(kernel, &beta)| {
                    crate::scaled_branching_term(
                        expected_marks[source],
                        alphas[alpha_index(target, source, kernel, component_count, betas.len())],
                        beta,
                    )
                })
                .sum();
        }
    }
    matrix
}

fn spectral_radius_upper_bound(matrix: &[f64], dimension: usize) -> HawkesResult<f64> {
    let mut vector = vec![1.0 / dimension as f64; dimension];
    let mut next = vec![0.0; dimension];
    let mut upper = f64::INFINITY;
    for _ in 0..1_024 {
        next.fill(0.0);
        for row in 0..dimension {
            next[row] = vector[row]
                + (0..dimension)
                    .map(|column| matrix[row * dimension + column] * vector[column])
                    .sum::<f64>();
        }
        if next.iter().any(|value| !value.is_finite() || *value <= 0.0) {
            return Err(HawkesError::InvalidBranchingRatio(f64::INFINITY));
        }
        let lower_shifted = next
            .iter()
            .zip(&vector)
            .map(|(&next_value, &value)| next_value / value)
            .fold(f64::INFINITY, f64::min);
        let upper_shifted = next
            .iter()
            .zip(&vector)
            .map(|(&next_value, &value)| next_value / value)
            .fold(0.0, f64::max);
        upper = (upper_shifted - 1.0).max(0.0);
        let lower = (lower_shifted - 1.0).max(0.0);
        if upper - lower <= 1.0e-12 * (1.0 + upper) {
            return Ok(upper);
        }
        let norm = next.iter().sum::<f64>();
        for (value, &next_value) in vector.iter_mut().zip(&next) {
            *value = next_value / norm;
        }
    }
    Ok(upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unstable_branching_matrix() {
        let error = MultivariateSumExpHawkes::new(
            vec![1.0, 1.0],
            vec![0.6, 0.0, 0.0, 0.6],
            vec![1.0],
            vec![2.0, 2.0],
        )
        .unwrap_err();
        assert!(matches!(error, HawkesError::InvalidBranchingRatio(value) if value >= 1.0));
    }

    #[test]
    fn batch_events_do_not_excite_each_other_before_commit() {
        let mut model = MultivariateSumExpHawkes::new(
            vec![1.0, 1.0],
            vec![0.2, 0.1, 0.1, 0.2],
            vec![1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        let events = [
            ComponentMark::new(0, 1.0).unwrap(),
            ComponentMark::new(1, 1.0).unwrap(),
        ];
        let intensities = model.update_batch(0, &events).unwrap();

        assert!((intensities[0] - 1.3).abs() < 1.0e-12);
        assert!((intensities[1] - 1.3).abs() < 1.0e-12);
    }

    #[test]
    fn online_batch_rejects_duplicate_components_without_committing() {
        let mut model = MultivariateSumExpHawkes::new(
            vec![1.0, 1.0],
            vec![0.2, 0.1, 0.1, 0.2],
            vec![1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        let duplicate = [
            ComponentMark::new(0, 1.0).unwrap(),
            ComponentMark::new(0, 2.0).unwrap(),
        ];

        assert!(matches!(
            model.update_batch(1_000, &duplicate),
            Err(HawkesError::DuplicateComponentAtTimestamp { component: 0, .. })
        ));
        assert_eq!(model.last_timestamp_us(), None);
        assert!(model.excitations.iter().all(|&value| value == 0.0));

        let valid = [ComponentMark::new(0, 1.0).unwrap()];
        assert!(model.update_batch(1_000, &valid).is_ok());
    }

    #[test]
    fn online_batch_rejects_successive_batches_at_one_timestamp() {
        let mut model = MultivariateSumExpHawkes::new(
            vec![1.0, 1.0],
            vec![0.2, 0.1, 0.1, 0.2],
            vec![1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        model
            .update_batch(1_000, &[ComponentMark::new(0, 1.0).unwrap()])
            .unwrap();
        let excitations = model.excitations.clone();

        assert_eq!(
            model
                .update_batch(1_000, &[ComponentMark::new(1, 1.0).unwrap()])
                .unwrap_err(),
            HawkesError::DuplicateTimestamp(0.001)
        );
        assert_eq!(model.excitations, excitations);
    }

    #[test]
    fn serialization_restores_runtime_state() {
        let mut model = MultivariateSumExpHawkes::new(
            vec![1.0, 1.0],
            vec![0.2, 0.1, 0.1, 0.2],
            vec![1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        model
            .update_batch(0, &[ComponentMark::new(0, 1.0).unwrap()])
            .unwrap();
        let encoded = serde_json::to_string(&model).unwrap();
        let decoded: MultivariateSumExpHawkes = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded.current_intensities(), model.current_intensities());
        assert_eq!(decoded.last_timestamp_us(), Some(0));
    }

    #[test]
    fn public_multivariate_fit_is_stationary() {
        let events = (0..30)
            .flat_map(|index| {
                (0..2).map(move |component| {
                    MultivariateEvent::new(index as f64 * 0.1, component, 1.0).unwrap()
                })
            })
            .collect::<Vec<_>>();
        let shape = PeriodicShape::new(2.0, vec![1.0, 1.0]).unwrap();
        let model = MultivariateSumExpHawkes::fit_marked_seasonal(
            events,
            2,
            vec![2.0, 0.5],
            vec![1.0, 1.0],
            vec![shape.clone(), shape],
        )
        .unwrap();

        assert!(model.branching_spectral_radius_upper_bound() < 1.0);
        assert_eq!(model.branching_matrix().len(), 4);
    }

    #[test]
    fn rejects_duplicate_component_within_timestamp_batch() {
        let events = [
            MultivariateEvent::new(0.0, 0, 1.0).unwrap(),
            MultivariateEvent::new(0.0, 0, 2.0).unwrap(),
            MultivariateEvent::new(1.0, 1, 1.0).unwrap(),
        ];
        assert!(matches!(
            validate_events(&events, 2),
            Err(HawkesError::DuplicateComponentAtTimestamp { component: 0, .. })
        ));
    }

    #[test]
    fn baseline_recalibration_matches_constant_poisson_mle() {
        let model =
            MultivariateSumExpHawkes::new(vec![2.0, 3.0], vec![0.0; 4], vec![1.0], vec![1.0, 1.0])
                .unwrap();
        let events = (0..4)
            .flat_map(|timestamp| {
                (0..2).map(move |component| {
                    MultivariateEvent::new(timestamp as f64, component, 1.0).unwrap()
                })
            })
            .collect::<Vec<_>>();

        let calibrated = model.recalibrate_baselines(&events, 0.0).unwrap();

        assert!((calibrated.mus()[0] - 1.0).abs() < 1.0e-12);
        assert!((calibrated.mus()[1] - 1.0).abs() < 1.0e-12);
        assert_eq!(calibrated.branching_matrix(), model.branching_matrix());
        assert_eq!(calibrated.last_timestamp_us(), None);
    }

    #[test]
    fn baseline_recalibration_validates_floor() {
        let model =
            MultivariateSumExpHawkes::new(vec![1.0], vec![0.0], vec![1.0], vec![1.0]).unwrap();
        let events = [
            MultivariateEvent::new(0.0, 0, 1.0).unwrap(),
            MultivariateEvent::new(1.0, 0, 1.0).unwrap(),
        ];

        assert!(model.recalibrate_baselines(&events, 1.0).is_err());
    }
}
