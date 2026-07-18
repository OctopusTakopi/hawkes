//! Constrained maximum-likelihood fitting for multivariate Hawkes models.

use crate::multivariate::{MultivariateEvent, MultivariateFitConfig, alpha_index, validate_events};
use crate::{HawkesError, HawkesResult, PeriodicShape};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::quasinewton::LBFGS;

struct MultivariateLikelihood {
    component_count: usize,
    fixed_betas: Vec<f64>,
    expected_marks: Vec<f64>,
    event_components: Vec<usize>,
    event_histories: Vec<f64>,
    integral_sums: Vec<f64>,
    baseline_multipliers: Vec<f64>,
    baseline_integrals: Vec<f64>,
    conditioning_event_count: usize,
    scored_counts: Vec<usize>,
    min_mus: Vec<f64>,
    max_source_offspring: f64,
    max_iterations: u64,
    duration: f64,
}

impl MultivariateLikelihood {
    fn new(
        events: Vec<MultivariateEvent>,
        component_count: usize,
        fixed_betas: Vec<f64>,
        expected_marks: Vec<f64>,
        seasonal_shapes: &[PeriodicShape],
        config: MultivariateFitConfig,
    ) -> HawkesResult<Self> {
        validate_inputs(
            &events,
            component_count,
            &fixed_betas,
            &expected_marks,
            seasonal_shapes,
        )?;
        let conditioning_event_count = events
            .iter()
            .take_while(|event| event.timestamp_seconds() == events[0].timestamp_seconds())
            .count();
        if conditioning_event_count == events.len() {
            return Err(HawkesError::InvalidObservationWindow);
        }
        let start = events[0].timestamp_seconds();
        let end = events[events.len() - 1].timestamp_seconds();
        let duration = end - start;
        let baseline_multipliers = events
            .iter()
            .map(|event| seasonal_shapes[event.component()].multiplier(event.timestamp_seconds()))
            .collect::<HawkesResult<Vec<_>>>()?;
        let baseline_integrals = seasonal_shapes
            .iter()
            .map(|shape| shape.integrated_multiplier(start, end))
            .collect::<HawkesResult<Vec<_>>>()?;
        let (event_histories, integral_sums) =
            compute_likelihood_cache(&events, component_count, &fixed_betas)?;
        let event_components = events.iter().map(MultivariateEvent::component).collect();
        let mut scored_counts = vec![0; component_count];
        for event in &events[conditioning_event_count..] {
            scored_counts[event.component()] += 1;
        }
        if let Some(component) = scored_counts.iter().position(|&count| count == 0) {
            return Err(HawkesError::FittingError(format!(
                "component {component} has no events after the conditioning batch"
            )));
        }
        let min_mus = scored_counts
            .iter()
            .map(|&count| config.min_baseline_fraction() * count as f64 / duration)
            .collect();

        Ok(Self {
            component_count,
            fixed_betas,
            expected_marks,
            event_components,
            event_histories,
            integral_sums,
            baseline_multipliers,
            baseline_integrals,
            conditioning_event_count,
            scored_counts,
            min_mus,
            max_source_offspring: config.max_source_offspring(),
            max_iterations: config.max_iterations(),
            duration,
        })
    }

    fn parameter_count(&self) -> usize {
        self.component_count + self.component_count * self.component_count * self.fixed_betas.len()
    }

    fn convert_params(&self, parameters: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        convert_params(
            parameters,
            self.component_count,
            &self.fixed_betas,
            &self.expected_marks,
            &self.min_mus,
            self.max_source_offspring,
        )
    }
}

fn validate_inputs(
    events: &[MultivariateEvent],
    component_count: usize,
    fixed_betas: &[f64],
    expected_marks: &[f64],
    seasonal_shapes: &[PeriodicShape],
) -> HawkesResult<()> {
    validate_events(events, component_count)?;
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
    if fixed_betas.is_empty() {
        return Err(HawkesError::FittingError(
            "fixed_betas must contain at least one decay rate".to_string(),
        ));
    }
    component_count
        .checked_mul(component_count)
        .and_then(|value| value.checked_mul(fixed_betas.len()))
        .ok_or_else(|| HawkesError::FittingError("alpha tensor size overflow".to_string()))?;
    for &beta in fixed_betas {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(if beta.is_finite() {
                HawkesError::InvalidBeta(beta)
            } else {
                HawkesError::NonFiniteParameter("beta", beta)
            });
        }
    }
    for &expected_mark in expected_marks {
        if !expected_mark.is_finite() || expected_mark <= 0.0 {
            return Err(HawkesError::InvalidFittingExpectedVolume(expected_mark));
        }
    }
    Ok(())
}

fn compute_likelihood_cache(
    events: &[MultivariateEvent],
    component_count: usize,
    fixed_betas: &[f64],
) -> HawkesResult<(Vec<f64>, Vec<f64>)> {
    let kernel_count = fixed_betas.len();
    let history_width = component_count.checked_mul(kernel_count).ok_or_else(|| {
        HawkesError::FittingError("multivariate history width overflow".to_string())
    })?;
    let cache_len = events.len().checked_mul(history_width).ok_or_else(|| {
        HawkesError::FittingError("multivariate likelihood cache size overflow".to_string())
    })?;
    let mut histories = Vec::new();
    histories
        .try_reserve_exact(cache_len)
        .map_err(|error| HawkesError::FittingError(error.to_string()))?;
    histories.resize(cache_len, 0.0);
    let mut recursion = vec![0.0; history_width];
    let mut integral_sums = vec![0.0; history_width];
    let mut batch_start = 0;
    let mut previous_timestamp = events[0].timestamp_seconds();

    while batch_start < events.len() {
        let timestamp = events[batch_start].timestamp_seconds();
        if batch_start > 0 {
            let delta = timestamp - previous_timestamp;
            for source in 0..component_count {
                for (kernel, &beta) in fixed_betas.iter().enumerate() {
                    let index = source * kernel_count + kernel;
                    let decay = (-beta * delta).exp();
                    integral_sums[index] += recursion[index] * (1.0 - decay);
                    recursion[index] *= decay;
                }
            }
        }

        let batch_end = events[batch_start..]
            .iter()
            .position(|event| event.timestamp_seconds() != timestamp)
            .map_or(events.len(), |offset| batch_start + offset);
        for event_index in batch_start..batch_end {
            histories[event_index * history_width..(event_index + 1) * history_width]
                .copy_from_slice(&recursion);
        }
        for event in &events[batch_start..batch_end] {
            let source_offset = event.component() * kernel_count;
            for value in &mut recursion[source_offset..source_offset + kernel_count] {
                *value += event.mark();
            }
        }

        previous_timestamp = timestamp;
        batch_start = batch_end;
    }

    if histories.iter().any(|value| !value.is_finite())
        || integral_sums.iter().any(|value| !value.is_finite())
    {
        return Err(HawkesError::FittingError(
            "multivariate likelihood cache overflowed".to_string(),
        ));
    }
    Ok((histories, integral_sums))
}

fn convert_params(
    parameters: &[f64],
    component_count: usize,
    fixed_betas: &[f64],
    expected_marks: &[f64],
    min_mus: &[f64],
    max_source_offspring: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let alpha_count = component_count * component_count * fixed_betas.len();
    if parameters.len() != component_count + alpha_count
        || parameters.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    if min_mus.len() != component_count {
        return None;
    }
    let mus = parameters[..component_count]
        .iter()
        .zip(min_mus)
        .map(|(value, &min_mu)| min_mu + value.exp())
        .collect::<Vec<_>>();
    if mus.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return None;
    }

    let kernel_count = fixed_betas.len();
    let mut alphas = vec![0.0; alpha_count];
    for source in 0..component_count {
        let max_raw = (0..component_count)
            .flat_map(|target| {
                (0..kernel_count).map(move |kernel| {
                    parameters[component_count
                        + alpha_index(target, source, kernel, component_count, kernel_count)]
                })
            })
            .fold(0.0, f64::max);
        let partition = (-max_raw).exp()
            + (0..component_count)
                .flat_map(|target| {
                    (0..kernel_count).map(move |kernel| {
                        let index =
                            alpha_index(target, source, kernel, component_count, kernel_count);
                        (parameters[component_count + index] - max_raw).exp()
                    })
                })
                .sum::<f64>();

        for target in 0..component_count {
            for (kernel, &beta) in fixed_betas.iter().enumerate() {
                let index = alpha_index(target, source, kernel, component_count, kernel_count);
                alphas[index] = beta
                    * max_source_offspring
                    * (parameters[component_count + index] - max_raw).exp()
                    / (expected_marks[source] * partition);
            }
        }
    }
    if alphas.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some((mus, alphas))
}

impl CostFunction for MultivariateLikelihood {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, parameters: &Self::Param) -> Result<Self::Output, Error> {
        let Some((mus, alphas)) = self.convert_params(parameters) else {
            return Ok(f64::INFINITY);
        };
        let kernel_count = self.fixed_betas.len();
        let history_width = self.component_count * kernel_count;
        let mut log_intensity = 0.0;

        for event_index in self.conditioning_event_count..self.event_components.len() {
            let target = self.event_components[event_index];
            let history = &self.event_histories
                [event_index * history_width..(event_index + 1) * history_width];
            let excitation = (0..self.component_count)
                .flat_map(|source| (0..kernel_count).map(move |kernel| (source, kernel)))
                .map(|(source, kernel)| {
                    alphas[alpha_index(target, source, kernel, self.component_count, kernel_count)]
                        * history[source * kernel_count + kernel]
                })
                .sum::<f64>();
            let intensity = mus[target] * self.baseline_multipliers[event_index] + excitation;
            log_intensity += intensity.ln();
        }

        let baseline_integral = mus
            .iter()
            .zip(&self.baseline_integrals)
            .map(|(&mu, &integral)| mu * integral)
            .sum::<f64>();
        let excitation_integral = (0..self.component_count)
            .flat_map(|target| {
                (0..self.component_count).flat_map(move |source| {
                    (0..kernel_count).map(move |kernel| (target, source, kernel))
                })
            })
            .map(|(target, source, kernel)| {
                alphas[alpha_index(target, source, kernel, self.component_count, kernel_count)]
                    * self.integral_sums[source * kernel_count + kernel]
                    / self.fixed_betas[kernel]
            })
            .sum::<f64>();

        Ok(-(log_intensity - baseline_integral - excitation_integral))
    }
}

impl Gradient for MultivariateLikelihood {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, parameters: &Self::Param) -> Result<Self::Gradient, Error> {
        let (mus, alphas) = self
            .convert_params(parameters)
            .ok_or_else(|| Error::msg("parameters out of bounds"))?;
        let kernel_count = self.fixed_betas.len();
        let history_width = self.component_count * kernel_count;
        let alpha_count = alphas.len();
        let mut grad_mu = self.baseline_integrals.to_vec();
        let mut grad_alpha = vec![0.0; alpha_count];
        for target in 0..self.component_count {
            for source in 0..self.component_count {
                for kernel in 0..kernel_count {
                    grad_alpha
                        [alpha_index(target, source, kernel, self.component_count, kernel_count)] =
                        self.integral_sums[source * kernel_count + kernel]
                            / self.fixed_betas[kernel];
                }
            }
        }

        for event_index in self.conditioning_event_count..self.event_components.len() {
            let target = self.event_components[event_index];
            let history = &self.event_histories
                [event_index * history_width..(event_index + 1) * history_width];
            let excitation = (0..self.component_count)
                .flat_map(|source| (0..kernel_count).map(move |kernel| (source, kernel)))
                .map(|(source, kernel)| {
                    alphas[alpha_index(target, source, kernel, self.component_count, kernel_count)]
                        * history[source * kernel_count + kernel]
                })
                .sum::<f64>();
            let baseline_multiplier = self.baseline_multipliers[event_index];
            let inverse_intensity = 1.0 / (mus[target] * baseline_multiplier + excitation);
            grad_mu[target] -= baseline_multiplier * inverse_intensity;
            for source in 0..self.component_count {
                for kernel in 0..kernel_count {
                    let index =
                        alpha_index(target, source, kernel, self.component_count, kernel_count);
                    grad_alpha[index] -=
                        history[source * kernel_count + kernel] * inverse_intensity;
                }
            }
        }

        let mut gradient = Vec::with_capacity(self.parameter_count());
        gradient.extend(
            grad_mu
                .iter()
                .zip(mus.iter().zip(&self.min_mus))
                .map(|(&gradient, (&mu, &min_mu))| gradient * (mu - min_mu)),
        );
        for source in 0..self.component_count {
            let weighted = (0..self.component_count)
                .flat_map(|target| {
                    (0..kernel_count).map(move |kernel| {
                        alpha_index(target, source, kernel, self.component_count, kernel_count)
                    })
                })
                .map(|index| grad_alpha[index] * alphas[index])
                .sum::<f64>();
            for target in 0..self.component_count {
                for kernel in 0..kernel_count {
                    let index =
                        alpha_index(target, source, kernel, self.component_count, kernel_count);
                    gradient.push(
                        alphas[index]
                            * (grad_alpha[index]
                                - self.expected_marks[source] * weighted
                                    / (self.max_source_offspring * self.fixed_betas[kernel])),
                    );
                }
            }
        }

        // `gradient` currently groups raw alpha parameters by source. Reorder
        // them into the public target/source/kernel tensor layout.
        let mut ordered = vec![0.0; self.parameter_count()];
        ordered[..self.component_count].copy_from_slice(&gradient[..self.component_count]);
        let mut grouped_index = self.component_count;
        for source in 0..self.component_count {
            for target in 0..self.component_count {
                for kernel in 0..kernel_count {
                    let tensor_index =
                        alpha_index(target, source, kernel, self.component_count, kernel_count);
                    ordered[self.component_count + tensor_index] = gradient[grouped_index];
                    grouped_index += 1;
                }
            }
        }
        Ok(ordered)
    }
}

pub(crate) fn fit_marked_seasonal(
    events: Vec<MultivariateEvent>,
    component_count: usize,
    fixed_betas: Vec<f64>,
    expected_marks: Vec<f64>,
    seasonal_shapes: &[PeriodicShape],
    config: MultivariateFitConfig,
) -> HawkesResult<(Vec<f64>, Vec<f64>)> {
    let likelihood = MultivariateLikelihood::new(
        events,
        component_count,
        fixed_betas,
        expected_marks,
        seasonal_shapes,
        config,
    )?;
    fit_likelihood(likelihood)
}

fn fit_likelihood(likelihood: MultivariateLikelihood) -> HawkesResult<(Vec<f64>, Vec<f64>)> {
    let target_min_mus = likelihood.min_mus.clone();
    let has_baseline_floor = target_min_mus.iter().any(|&min_mu| min_mu > 0.0);
    let (likelihood, parameters) = if has_baseline_floor {
        let mut unconstrained = likelihood;
        unconstrained.min_mus.fill(0.0);
        let initial = initial_parameters(&unconstrained);
        let (mut constrained, mut parameters) = optimize_likelihood(unconstrained, initial)?;
        constrained.min_mus = target_min_mus;
        for (component, parameter) in parameters[..constrained.component_count]
            .iter_mut()
            .enumerate()
        {
            let empirical_rate = constrained.scored_counts[component] as f64 / constrained.duration;
            let previous_mu = parameter.exp();
            *parameter = (previous_mu - constrained.min_mus[component])
                .max(empirical_rate * 1.0e-6)
                .ln();
        }
        optimize_likelihood(constrained, parameters)?
    } else {
        let initial = initial_parameters(&likelihood);
        optimize_likelihood(likelihood, initial)?
    };
    convert_params(
        &parameters,
        likelihood.component_count,
        &likelihood.fixed_betas,
        &likelihood.expected_marks,
        &likelihood.min_mus,
        likelihood.max_source_offspring,
    )
    .ok_or_else(|| HawkesError::FittingError("optimizer returned invalid parameters".to_string()))
}

fn initial_parameters(likelihood: &MultivariateLikelihood) -> Vec<f64> {
    let mut initial = likelihood
        .scored_counts
        .iter()
        .zip(&likelihood.min_mus)
        .map(|(&count, &min_mu)| {
            let empirical_rate = count as f64 / likelihood.duration;
            (0.8 * empirical_rate - min_mu)
                .max(empirical_rate * 1.0e-6)
                .ln()
        })
        .collect::<Vec<_>>();
    let initial_offspring = 0.2_f64.min(0.5 * likelihood.max_source_offspring);
    let raw_mass = initial_offspring / (likelihood.max_source_offspring - initial_offspring);
    let raw_branch =
        (raw_mass / (likelihood.component_count * likelihood.fixed_betas.len()) as f64).ln();
    initial.resize(likelihood.parameter_count(), raw_branch);
    initial
}

fn optimize_likelihood(
    likelihood: MultivariateLikelihood,
    initial: Vec<f64>,
) -> HawkesResult<(MultivariateLikelihood, Vec<f64>)> {
    let armijo = ArmijoCondition::new(0.0001)
        .map_err(|error| HawkesError::FittingError(error.to_string()))?;
    let solver = LBFGS::new(BacktrackingLineSearch::new(armijo), 10)
        .with_tolerance_grad(1.0e-4)
        .map_err(|error| HawkesError::FittingError(error.to_string()))?
        .with_tolerance_cost(1.0e-4)
        .map_err(|error| HawkesError::FittingError(error.to_string()))?;
    let max_iterations = likelihood.max_iterations;
    let mut result = Executor::new(likelihood, solver)
        .configure(|state| state.param(initial).max_iters(max_iterations))
        .run()
        .map_err(|error| HawkesError::FittingError(error.to_string()))?;
    let parameters = result
        .state
        .best_param
        .ok_or_else(|| HawkesError::FittingError("no best parameter found".to_string()))?;
    let likelihood = result
        .problem
        .take_problem()
        .ok_or_else(|| HawkesError::FittingError("optimizer lost likelihood state".to_string()))?;
    Ok((likelihood, parameters))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn likelihood() -> MultivariateLikelihood {
        let events = vec![
            MultivariateEvent::new(0.0, 0, 1.0).unwrap(),
            MultivariateEvent::new(0.0, 1, 1.0).unwrap(),
            MultivariateEvent::new(0.2, 0, 0.5).unwrap(),
            MultivariateEvent::new(0.7, 1, 1.5).unwrap(),
            MultivariateEvent::new(1.4, 0, 1.0).unwrap(),
            MultivariateEvent::new(1.4, 1, 0.5).unwrap(),
        ];
        let shape = PeriodicShape::new(2.0, vec![0.5, 1.5]).unwrap();
        MultivariateLikelihood::new(
            events,
            2,
            vec![2.0, 0.5],
            vec![1.0, 1.0],
            &[shape.clone(), shape],
            MultivariateFitConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn gradient_matches_finite_difference_with_timestamp_ties() {
        let likelihood = likelihood();
        let parameters = vec![-0.3, 0.1, 0.2, -0.4, 0.3, -0.2, -0.1, 0.4, 0.1, -0.3];
        let analytical = likelihood.gradient(&parameters).unwrap();
        let epsilon = 1.0e-6;
        for (index, &analytical_value) in analytical.iter().enumerate() {
            let mut upper = parameters.clone();
            let mut lower = parameters.clone();
            upper[index] += epsilon;
            lower[index] -= epsilon;
            let numerical = (likelihood.cost(&upper).unwrap() - likelihood.cost(&lower).unwrap())
                / (2.0 * epsilon);
            assert!(
                (analytical_value - numerical).abs() < 2.0e-6,
                "parameter {index}: analytical {analytical_value}, numerical {numerical}"
            );
        }
    }

    #[test]
    fn parameterization_bounds_each_source_column() {
        let likelihood = likelihood();
        let parameters = vec![100.0; likelihood.parameter_count()];
        let (_, alphas) = likelihood.convert_params(&parameters).unwrap();
        for source in 0..likelihood.component_count {
            let mut offspring = 0.0;
            for target in 0..likelihood.component_count {
                for kernel in 0..likelihood.fixed_betas.len() {
                    offspring += alphas[alpha_index(
                        target,
                        source,
                        kernel,
                        likelihood.component_count,
                        likelihood.fixed_betas.len(),
                    )] / likelihood.fixed_betas[kernel];
                }
            }
            assert!(offspring <= likelihood.max_source_offspring);
            assert!(offspring < 1.0);
        }
    }
}
