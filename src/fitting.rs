//! Hawkes Process parameter fitting using Maximum Likelihood Estimation (MLE).
//!
//! This module provides the infrastructure to fit Hawkes Process parameters (mu, alphas)
//! to a sequence of timestamps using the L-BFGS optimization algorithm.

use crate::{HawkesError, HawkesResult, PeriodicShape};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::quasinewton::LBFGS;

/// Holds parameter-independent likelihood terms computed from immutable inputs.
struct HawkesLikelihood {
    fixed_betas: Vec<f64>,
    expected_volume: f64,
    integral_sums: Vec<f64>,
    event_recursions: Vec<f64>,
    baseline_multipliers: Vec<f64>,
    baseline_integral: f64,
    duration: f64,
    event_count: usize,
}

const MAX_BRANCHING_RATIO: f64 = 1.0 - 1.0e-8;

impl HawkesLikelihood {
    fn new(timestamps: Vec<f64>, fixed_betas: Vec<f64>) -> HawkesResult<Self> {
        let volumes = vec![1.0; timestamps.len()];
        Self::new_marked(timestamps, volumes, fixed_betas, 1.0)
    }

    fn new_marked(
        timestamps: Vec<f64>,
        volumes: Vec<f64>,
        fixed_betas: Vec<f64>,
        expected_volume: f64,
    ) -> HawkesResult<Self> {
        Self::new_marked_with_shape(timestamps, volumes, fixed_betas, expected_volume, None)
    }

    fn new_marked_with_shape(
        timestamps: Vec<f64>,
        volumes: Vec<f64>,
        fixed_betas: Vec<f64>,
        expected_volume: f64,
        seasonal_shape: Option<&PeriodicShape>,
    ) -> HawkesResult<Self> {
        validate_fit_inputs(&timestamps, &volumes, &fixed_betas, expected_volume)?;
        let (event_recursions, integral_sums) =
            compute_likelihood_cache(&timestamps, &volumes, &fixed_betas)?;
        let duration = timestamps[timestamps.len() - 1] - timestamps[0];
        let (baseline_multipliers, baseline_integral) = match seasonal_shape {
            Some(shape) => {
                let multipliers = timestamps
                    .iter()
                    .map(|&timestamp| shape.multiplier(timestamp))
                    .collect::<HawkesResult<Vec<_>>>()?;
                let integral =
                    shape.integrated_multiplier(timestamps[0], timestamps[timestamps.len() - 1])?;
                (multipliers, integral)
            }
            None => (vec![1.0; timestamps.len()], duration),
        };
        Ok(Self {
            fixed_betas,
            expected_volume,
            integral_sums,
            event_recursions,
            baseline_multipliers,
            baseline_integral,
            duration,
            event_count: timestamps.len(),
        })
    }

    fn integral_sums(&self) -> &[f64] {
        &self.integral_sums
    }
}

fn validate_fit_inputs(
    timestamps: &[f64],
    volumes: &[f64],
    fixed_betas: &[f64],
    expected_volume: f64,
) -> HawkesResult<()> {
    if timestamps.is_empty() {
        return Err(HawkesError::EmptyTimestamps);
    }

    if timestamps.iter().any(|t| !t.is_finite()) {
        return Err(HawkesError::FittingError(
            "timestamps must all be finite".to_string(),
        ));
    }

    if timestamps.windows(2).any(|w| w[1] < w[0]) {
        return Err(HawkesError::UnsortedTimestamps);
    }

    let duration = timestamps[timestamps.len() - 1] - timestamps[0];
    if !duration.is_finite() || duration <= 0.0 {
        return Err(HawkesError::InvalidObservationWindow);
    }

    if timestamps.len() != volumes.len() {
        return Err(HawkesError::VolumeDimensionMismatch(
            volumes.len(),
            timestamps.len(),
        ));
    }

    for &volume in volumes {
        if !volume.is_finite() {
            return Err(HawkesError::NonFiniteParameter("volume", volume));
        }
        if volume < 0.0 {
            return Err(HawkesError::InvalidVolume(volume));
        }
    }

    if !expected_volume.is_finite() {
        return Err(HawkesError::NonFiniteParameter(
            "expected_volume",
            expected_volume,
        ));
    }
    if expected_volume <= 0.0 {
        return Err(HawkesError::InvalidFittingExpectedVolume(expected_volume));
    }

    if fixed_betas.is_empty() {
        return Err(HawkesError::FittingError(
            "fixed_betas must contain at least one decay rate".to_string(),
        ));
    }

    if fixed_betas.iter().any(|beta| !beta.is_finite()) {
        return Err(HawkesError::FittingError(
            "fixed_betas must all be finite".to_string(),
        ));
    }

    if fixed_betas.iter().any(|&beta| beta <= 0.0) {
        return Err(HawkesError::FittingError(
            "fixed_betas must all be strictly positive".to_string(),
        ));
    }

    Ok(())
}

/// Computes decayed mark histories and kernel compensator sums in one pass.
///
/// The result is event-major: row `i` occupies `i * K..(i + 1) * K`.
/// It depends only on observations and fixed decay rates, so every optimizer
/// cost and gradient evaluation can reuse it without evaluating exponentials.
fn compute_likelihood_cache(
    timestamps: &[f64],
    volumes: &[f64],
    fixed_betas: &[f64],
) -> HawkesResult<(Vec<f64>, Vec<f64>)> {
    let kernel_count = fixed_betas.len();
    let cache_len = timestamps.len().checked_mul(kernel_count).ok_or_else(|| {
        HawkesError::FittingError("likelihood recursion cache size overflow".to_string())
    })?;
    let mut histories = Vec::new();
    histories
        .try_reserve_exact(cache_len)
        .map_err(|error| HawkesError::FittingError(error.to_string()))?;
    histories.resize(cache_len, 0.0);
    let mut recursion = vec![0.0; kernel_count];
    let mut integral_sums = vec![0.0; kernel_count];

    for event_index in 0..timestamps.len() {
        if event_index > 0 {
            let delta = timestamps[event_index] - timestamps[event_index - 1];
            let row = &mut histories[event_index * kernel_count..(event_index + 1) * kernel_count];
            for (((value, &beta), cached), integral_sum) in recursion
                .iter_mut()
                .zip(fixed_betas)
                .zip(row.iter_mut())
                .zip(&mut integral_sums)
            {
                let decay = (-beta * delta).exp();
                *integral_sum += *value * (1.0 - decay);
                *value *= decay;
                *cached = *value;
            }
        }

        for value in &mut recursion {
            *value += volumes[event_index];
        }
    }

    if histories.iter().any(|value| !value.is_finite())
        || integral_sums.iter().any(|value| !value.is_finite())
    {
        return Err(HawkesError::FittingError(
            "likelihood recursion cache overflowed".to_string(),
        ));
    }

    Ok((histories, integral_sums))
}

impl HawkesLikelihood {
    /// Computes NegLogLikelihood and Gradient.
    /// Returns (nll, grad)
    /// Params: [log_mu, raw_branch_0, raw_branch_1, ...]
    /// mu = exp(log_mu)
    /// alpha_k = rho_max * beta_k * exp(raw_branch_k)
    ///           / (E[V] * (1 + Σ_j exp(raw_branch_j)))
    /// where rho_max is a fixed, representable value below one.
    /// This guarantees E[V] * Σ_k alpha_k / beta_k < 1.
    fn convert_params(&self, p: &[f64]) -> Option<(f64, Vec<f64>)> {
        convert_params(p, &self.fixed_betas, self.expected_volume)
    }
}

fn convert_params(p: &[f64], fixed_betas: &[f64], expected_volume: f64) -> Option<(f64, Vec<f64>)> {
    if p.len() != 1 + fixed_betas.len() {
        return None;
    }
    if p.iter().any(|x| !x.is_finite()) {
        return None;
    }

    let mu = p[0].exp();
    if !mu.is_finite() || mu <= 0.0 {
        return None;
    }

    // Compute alpha_k / beta_k using a stable softmax over the branch logits
    // plus one extra "slack" component of weight 1.0. The fixed scale
    // preserves representable slack even when that component underflows.
    let max_raw_branch = p[1..].iter().copied().fold(0.0, f64::max);
    let shifted_partition = (-max_raw_branch).exp()
        + p[1..]
            .iter()
            .map(|&raw| (raw - max_raw_branch).exp())
            .sum::<f64>();

    let alphas: Vec<f64> = fixed_betas
        .iter()
        .zip(p[1..].iter())
        .map(|(&beta, &raw)| {
            beta * MAX_BRANCHING_RATIO * (raw - max_raw_branch).exp()
                / (expected_volume * shifted_partition)
        })
        .collect();

    if alphas.iter().any(|alpha| !alpha.is_finite()) {
        return None;
    }

    Some((mu, alphas))
}

impl CostFunction for HawkesLikelihood {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let (mu, alphas) = match self.convert_params(param) {
            Some(v) => v,
            None => return Ok(f64::INFINITY),
        };

        let k_kernels = self.fixed_betas.len();
        debug_assert!(self.event_count > 0, "validated in HawkesLikelihood::new");

        let mut log_sum_lambda = 0.0;

        // Row zero is conditioning history and is intentionally not scored.
        for (row, &baseline_multiplier) in self
            .event_recursions
            .chunks_exact(k_kernels)
            .zip(&self.baseline_multipliers)
            .skip(1)
        {
            let excitation = alphas
                .iter()
                .zip(row)
                .map(|(&alpha, &history)| alpha * history)
                .sum::<f64>();
            let lambda_t = mu * baseline_multiplier + excitation;
            log_sum_lambda += lambda_t.ln();
        }

        let integral_mu = mu * self.baseline_integral;

        let term2: f64 = self
            .integral_sums()
            .iter()
            .zip(alphas.iter().zip(self.fixed_betas.iter()))
            .map(|(sum_exp, (&alpha, &beta))| (alpha / beta) * sum_exp)
            .sum();

        Ok(-(log_sum_lambda - (integral_mu + term2)))
    }
}

impl Gradient for HawkesLikelihood {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        let (mu, alphas) = match self.convert_params(param) {
            Some(v) => v,
            // Return an error so L-BFGS gets a real failure signal instead of
            // a zero gradient (which would look like a stationary point and could
            // cause the optimizer to terminate early or break the line search).
            None => {
                return Err(Error::msg(
                    "Parameters out of bounds in gradient evaluation",
                ));
            }
        };

        let k_kernels = self.fixed_betas.len();
        debug_assert!(self.event_count > 0, "validated in HawkesLikelihood::new");

        let mut grad_mu_term1 = 0.0;
        let mut grad_alpha_term1 = vec![0.0; k_kernels];

        for (row, &baseline_multiplier) in self
            .event_recursions
            .chunks_exact(k_kernels)
            .zip(&self.baseline_multipliers)
            .skip(1)
        {
            let excitation = alphas
                .iter()
                .zip(row)
                .map(|(&alpha, &history)| alpha * history)
                .sum::<f64>();
            let lambda_t = mu * baseline_multiplier + excitation;
            let inv_lambda = 1.0 / lambda_t;

            grad_mu_term1 += baseline_multiplier * inv_lambda;
            for (gradient, &history) in grad_alpha_term1.iter_mut().zip(row) {
                *gradient += history * inv_lambda;
            }
        }

        let grad_integral_mu = self.baseline_integral;
        let mut grad_integral_alpha = vec![0.0; k_kernels];

        for (k, grad) in grad_integral_alpha.iter_mut().enumerate().take(k_kernels) {
            *grad = self.integral_sums()[k] / self.fixed_betas[k];
        }

        // Combine for NegLogLikelihood Gradient w.r.t actual params
        let mut grad_actual = Vec::with_capacity(param.len());
        grad_actual.push(-(grad_mu_term1 - grad_integral_mu));
        for k in 0..k_kernels {
            grad_actual.push(-(grad_alpha_term1[k] - grad_integral_alpha[k]));
        }

        // Apply chain rule for the constrained parameterization.
        let mut grad_log = Vec::with_capacity(param.len());

        grad_log.push(grad_actual[0] * mu);

        let weighted_alpha_grad: f64 = grad_actual[1..]
            .iter()
            .zip(alphas.iter())
            .map(|(grad, alpha)| grad * alpha)
            .sum();

        for k in 0..k_kernels {
            grad_log.push(
                alphas[k]
                    * (grad_actual[1 + k]
                        - self.expected_volume * weighted_alpha_grad
                            / (MAX_BRANCHING_RATIO * self.fixed_betas[k])),
            );
        }

        Ok(grad_log)
    }
}

pub(crate) fn fit_hawkes(
    timestamps: Vec<f64>,
    fixed_betas: Vec<f64>,
) -> HawkesResult<(f64, Vec<f64>)> {
    let cost = HawkesLikelihood::new(timestamps, fixed_betas)?;
    fit_likelihood(cost)
}

pub(crate) fn fit_marked_hawkes(
    timestamps: Vec<f64>,
    volumes: Vec<f64>,
    fixed_betas: Vec<f64>,
    expected_volume: f64,
) -> HawkesResult<(f64, Vec<f64>)> {
    let cost = HawkesLikelihood::new_marked(timestamps, volumes, fixed_betas, expected_volume)?;
    fit_likelihood(cost)
}

pub(crate) fn fit_seasonal_hawkes(
    timestamps: Vec<f64>,
    fixed_betas: Vec<f64>,
    seasonal_shape: &PeriodicShape,
) -> HawkesResult<(f64, Vec<f64>)> {
    let volumes = vec![1.0; timestamps.len()];
    let cost = HawkesLikelihood::new_marked_with_shape(
        timestamps,
        volumes,
        fixed_betas,
        1.0,
        Some(seasonal_shape),
    )?;
    fit_likelihood(cost)
}

pub(crate) fn fit_marked_seasonal_hawkes(
    timestamps: Vec<f64>,
    volumes: Vec<f64>,
    fixed_betas: Vec<f64>,
    expected_volume: f64,
    seasonal_shape: &PeriodicShape,
) -> HawkesResult<(f64, Vec<f64>)> {
    let cost = HawkesLikelihood::new_marked_with_shape(
        timestamps,
        volumes,
        fixed_betas,
        expected_volume,
        Some(seasonal_shape),
    )?;
    fit_likelihood(cost)
}

fn fit_likelihood(cost: HawkesLikelihood) -> HawkesResult<(f64, Vec<f64>)> {
    // Initial guess:
    // Scale mu to the empirical event rate so optimization is insensitive to
    // whether timestamps are measured in seconds, milliseconds, or larger units.
    // alpha_k / beta_k share is initialized to about 0.2 / K, keeping the
    // total branching ratio strictly below 1.
    let empirical_rate = if cost.duration > 0.0 {
        (cost.event_count - 1) as f64 / cost.duration
    } else {
        1.0
    };
    let mut init_param = vec![(0.8 * empirical_rate).ln()];
    let target_branch_component = (0.2 / cost.fixed_betas.len() as f64).ln();
    for _ in &cost.fixed_betas {
        init_param.push(target_branch_component);
    }

    let armijo = ArmijoCondition::new(0.0001)
        .map_err(|error| HawkesError::FittingError(error.to_string()))?;
    let linesearch = BacktrackingLineSearch::new(armijo);
    let solver = LBFGS::new(linesearch, 7)
        .with_tolerance_grad(1e-4)
        .map_err(|error| HawkesError::FittingError(error.to_string()))?
        .with_tolerance_cost(1e-4)
        .map_err(|error| HawkesError::FittingError(error.to_string()))?;

    let output_betas = cost.fixed_betas.clone();
    let output_expected_volume = cost.expected_volume;
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .run()
        .map_err(|error| HawkesError::FittingError(error.to_string()))?;

    // Convert back from log params
    let best_param_log = res
        .state
        .best_param
        .ok_or_else(|| HawkesError::FittingError("no best parameter found".to_string()))?;
    let (mu, alphas) = convert_params(&best_param_log, &output_betas, output_expected_volume)
        .ok_or_else(|| {
            HawkesError::FittingError("optimizer returned invalid parameters".to_string())
        })?;

    Ok((mu, alphas))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_rejects_empty_timestamps() {
        let err = HawkesLikelihood::new(vec![], vec![1.0]).err().unwrap();
        assert!(matches!(err, HawkesError::EmptyTimestamps));
    }

    #[test]
    fn test_fit_rejects_unsorted_timestamps() {
        let err = HawkesLikelihood::new(vec![0.0, 2.0, 1.0], vec![1.0])
            .err()
            .unwrap();
        assert!(matches!(err, HawkesError::UnsortedTimestamps));
    }

    #[test]
    fn test_fit_requires_positive_observation_window() {
        assert!(matches!(
            HawkesLikelihood::new(vec![1.0], vec![1.0]),
            Err(HawkesError::InvalidObservationWindow)
        ));
        assert!(matches!(
            HawkesLikelihood::new(vec![1.0, 1.0], vec![1.0]),
            Err(HawkesError::InvalidObservationWindow)
        ));
    }

    #[test]
    fn test_precomputed_likelihood_cache_matches_reference() {
        let timestamps = vec![0.0, 1.0, 3.0];
        let volumes = vec![1.0; timestamps.len()];
        let betas = vec![2.0, 0.5];
        let (_, expected) = compute_likelihood_cache(&timestamps, &volumes, &betas).unwrap();
        let likelihood = HawkesLikelihood::new(timestamps, betas).unwrap();

        assert_eq!(likelihood.integral_sums(), expected.as_slice());
    }

    #[test]
    fn test_precomputed_event_recursions_match_reference() {
        let timestamps = vec![0.0, 0.5, 1.5];
        let volumes = vec![2.0, 3.0, 4.0];
        let betas = vec![1.0, 2.0];
        let (cached, integral_sums) =
            compute_likelihood_cache(&timestamps, &volumes, &betas).unwrap();
        let first_decay = (-0.5_f64).exp();
        let second_decay = (-1.0_f64).exp();

        assert_eq!(&cached[..2], &[0.0, 0.0]);
        assert!((cached[2] - 2.0 * first_decay).abs() < 1.0e-12);
        assert!((cached[3] - 2.0 * second_decay).abs() < 1.0e-12);
        assert!((cached[4] - (2.0 * first_decay + 3.0) * second_decay).abs() < 1.0e-12);
        assert!((cached[5] - (2.0 * second_decay + 3.0) * second_decay.powi(2)).abs() < 1.0e-12);
        for (kernel, &beta) in betas.iter().enumerate() {
            let expected = timestamps
                .iter()
                .zip(&volumes)
                .map(|(&timestamp, &volume)| {
                    volume * (1.0 - (-beta * (timestamps[2] - timestamp)).exp())
                })
                .sum::<f64>();
            assert!((integral_sums[kernel] - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_parameter_conversion_enforces_stationarity() {
        let likelihood = HawkesLikelihood::new(vec![0.0, 1.0, 3.0], vec![10.0, 1.0, 0.1]).unwrap();
        let (_mu, alphas) = likelihood.convert_params(&[0.0, 20.0, 20.0, 20.0]).unwrap();
        let branching_ratio: f64 = alphas
            .iter()
            .zip(likelihood.fixed_betas.iter())
            .map(|(&alpha, &beta)| alpha / beta)
            .sum();

        assert!(branching_ratio < 1.0);
    }

    #[test]
    fn test_parameter_conversion_preserves_boundary_slack() {
        let likelihood = HawkesLikelihood::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
        let (mu, alphas) = likelihood.convert_params(&[0.0, 1.0e308, 1.0e308]).unwrap();
        let branching_ratio: f64 = alphas
            .iter()
            .zip(&likelihood.fixed_betas)
            .map(|(&alpha, &beta)| alpha / beta)
            .sum();

        assert!(branching_ratio <= MAX_BRANCHING_RATIO);
        assert!(branching_ratio < 1.0);
        assert!(crate::SumExpHawkes::new(mu, alphas, likelihood.fixed_betas.clone()).is_ok());
    }

    #[test]
    fn test_constrained_gradient_matches_finite_difference() {
        let likelihood = HawkesLikelihood::new(vec![0.0, 0.2, 0.7, 1.4], vec![0.5, 2.0]).unwrap();
        let params = vec![-0.3, 0.4, -0.8];
        let analytical = likelihood.gradient(&params).unwrap();
        let epsilon = 1.0e-6;

        for (index, &analytical_value) in analytical.iter().enumerate() {
            let mut upper = params.clone();
            let mut lower = params.clone();
            upper[index] += epsilon;
            lower[index] -= epsilon;
            let numerical = (likelihood.cost(&upper).unwrap() - likelihood.cost(&lower).unwrap())
                / (2.0 * epsilon);

            assert!((analytical_value - numerical).abs() < 1.0e-6);
        }
    }

    #[test]
    fn test_marked_gradient_matches_finite_difference() {
        let likelihood = HawkesLikelihood::new_marked(
            vec![0.0, 0.2, 0.7, 1.4],
            vec![0.5, 2.0, 0.25, 1.25],
            vec![0.5, 2.0],
            1.0,
        )
        .unwrap();
        let params = vec![-0.3, 0.4, -0.8];
        let analytical = likelihood.gradient(&params).unwrap();
        let epsilon = 1.0e-6;

        for (index, &analytical_value) in analytical.iter().enumerate() {
            let mut upper = params.clone();
            let mut lower = params.clone();
            upper[index] += epsilon;
            lower[index] -= epsilon;
            let numerical = (likelihood.cost(&upper).unwrap() - likelihood.cost(&lower).unwrap())
                / (2.0 * epsilon);

            assert!((analytical_value - numerical).abs() < 1.0e-6);
        }
    }

    #[test]
    fn test_seasonal_gradient_matches_finite_difference() {
        let shape = PeriodicShape::new(2.0, vec![0.5, 1.5]).unwrap();
        let likelihood = HawkesLikelihood::new_marked_with_shape(
            vec![0.0, 0.2, 0.7, 1.4],
            vec![0.5, 2.0, 0.25, 1.25],
            vec![0.5, 2.0],
            1.0,
            Some(&shape),
        )
        .unwrap();
        let params = vec![-0.3, 0.4, -0.8];
        let analytical = likelihood.gradient(&params).unwrap();
        let epsilon = 1.0e-6;

        for (index, &analytical_value) in analytical.iter().enumerate() {
            let mut upper = params.clone();
            let mut lower = params.clone();
            upper[index] += epsilon;
            lower[index] -= epsilon;
            let numerical = (likelihood.cost(&upper).unwrap() - likelihood.cost(&lower).unwrap())
                / (2.0 * epsilon);

            assert!((analytical_value - numerical).abs() < 1.0e-6);
        }
    }

    #[test]
    fn test_likelihood_conditions_on_first_event() {
        let likelihood = HawkesLikelihood::new(vec![0.0, 1.0, 2.0], vec![1.0]).unwrap();
        let cost = likelihood.cost(&vec![2.0_f64.ln(), -1_000.0]).unwrap();
        let expected = 4.0 - 2.0 * 2.0_f64.ln();

        assert!((cost - expected).abs() < 1.0e-12);
    }

    #[test]
    fn test_marked_fit_validates_volume_inputs() {
        assert!(matches!(
            HawkesLikelihood::new_marked(vec![0.0, 1.0], vec![1.0], vec![1.0], 1.0),
            Err(HawkesError::VolumeDimensionMismatch(1, 2))
        ));
        assert!(matches!(
            HawkesLikelihood::new_marked(vec![0.0, 1.0], vec![1.0, 1.0], vec![1.0], 0.0),
            Err(HawkesError::InvalidFittingExpectedVolume(0.0))
        ));
        assert!(matches!(
            HawkesLikelihood::new_marked(
                vec![0.0, 1.0, 2.0],
                vec![f64::MAX, f64::MAX, 1.0],
                vec![1.0],
                1.0,
            ),
            Err(HawkesError::FittingError(message))
                if message == "likelihood recursion cache overflowed"
        ));
    }
}
