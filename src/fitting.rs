//! Hawkes Process parameter fitting using Maximum Likelihood Estimation (MLE).
//!
//! This module provides the infrastructure to fit Hawkes Process parameters (mu, alphas)
//! to a sequence of timestamps using the L-BFGS optimization algorithm.

use crate::{HawkesError, HawkesResult};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::quasinewton::LBFGS;
use rayon::prelude::*;

/// Holds the data and fixed configuration for ensuring the cost function is stateless.
#[derive(Clone)]
pub struct HawkesLikelihood {
    pub timestamps: Vec<f64>,  // Trade timestamps in seconds, sorted
    pub fixed_betas: Vec<f64>, // Fixed decay rates
    integral_sums: Vec<f64>,
}

impl HawkesLikelihood {
    pub fn new(timestamps: Vec<f64>, fixed_betas: Vec<f64>) -> HawkesResult<Self> {
        validate_fit_inputs(&timestamps, &fixed_betas)?;
        let integral_sums = compute_integral_sums(&timestamps, &fixed_betas);
        Ok(Self {
            timestamps,
            fixed_betas,
            integral_sums,
        })
    }

    fn integral_sums(&self) -> &[f64] {
        &self.integral_sums
    }
}

fn validate_fit_inputs(timestamps: &[f64], fixed_betas: &[f64]) -> HawkesResult<()> {
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

/// Computes Σ_i [1 - exp(-β_k · (t_max - t_i))] for every kernel k.
/// This term depends only on data and fixed betas, so it can be precomputed once.
fn compute_integral_sums(timestamps: &[f64], fixed_betas: &[f64]) -> Vec<f64> {
    let t_max = timestamps.last().copied().unwrap_or(0.0);
    let k_kernels = fixed_betas.len();

    timestamps
        .par_iter()
        .fold(
            || vec![0.0; k_kernels],
            |mut acc, &t| {
                for (k, acc_k) in acc.iter_mut().enumerate() {
                    *acc_k += 1.0 - (-fixed_betas[k] * (t_max - t)).exp();
                }
                acc
            },
        )
        .reduce(
            || vec![0.0; k_kernels],
            |mut a, b| {
                for (a_k, b_k) in a.iter_mut().zip(b.iter()) {
                    *a_k += b_k;
                }
                a
            },
        )
}

impl HawkesLikelihood {
    /// Computes NegLogLikelihood and Gradient.
    /// Returns (nll, grad)
    /// Params: [log_mu, raw_branch_0, raw_branch_1, ...]
    /// mu = exp(log_mu)
    /// alpha_k = beta_k * exp(raw_branch_k) / (1 + Σ_j exp(raw_branch_j))
    /// This guarantees Σ_k alpha_k / beta_k < 1, so fitted models stay stationary.
    pub fn convert_params(&self, p: &[f64]) -> Option<(f64, Vec<f64>)> {
        if p.len() != 1 + self.fixed_betas.len() {
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
        // plus one extra "slack" component of weight 1.0, which keeps the
        // branching ratio strictly below 1.
        let max_raw_branch = p[1..].iter().copied().fold(0.0, f64::max);
        let shifted_partition = (-max_raw_branch).exp()
            + p[1..]
                .iter()
                .map(|&raw| (raw - max_raw_branch).exp())
                .sum::<f64>();
        let log_denom = max_raw_branch + shifted_partition.ln();

        let alphas: Vec<f64> = self
            .fixed_betas
            .iter()
            .zip(p[1..].iter())
            .map(|(&beta, &raw)| beta * (raw - log_denom).exp())
            .collect();

        Some((mu, alphas))
    }
}

impl CostFunction for HawkesLikelihood {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let (mu, alphas) = match self.convert_params(param) {
            Some(v) => v,
            None => return Ok(f64::INFINITY),
        };

        let n = self.timestamps.len();
        let k_kernels = self.fixed_betas.len();
        debug_assert!(n > 0, "validated in HawkesLikelihood::new");

        let mut log_sum_lambda = 0.0;
        let mut r = vec![0.0; k_kernels];

        // i=0
        let lambda_0 = mu;
        log_sum_lambda += lambda_0.ln();
        for r_k in r.iter_mut() {
            *r_k += 1.0;
        }

        for i in 1..n {
            let delta_t = self.timestamps[i] - self.timestamps[i - 1];

            let mut excitation = 0.0;
            for k in 0..k_kernels {
                r[k] *= (-self.fixed_betas[k] * delta_t).exp();
                excitation += alphas[k] * r[k];
            }

            let lambda_t = mu + excitation;
            log_sum_lambda += lambda_t.ln();

            for r_k in r.iter_mut() {
                *r_k += 1.0;
            }
        }

        let t_max = self.timestamps.last().copied().unwrap_or(0.0);
        let t_min = self.timestamps.first().copied().unwrap_or(0.0);
        let total_duration = t_max - t_min;

        let integral_mu = mu * total_duration;

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

        let n = self.timestamps.len();
        let k_kernels = self.fixed_betas.len();
        debug_assert!(n > 0, "validated in HawkesLikelihood::new");

        let mut grad_mu_term1 = 0.0;
        let mut grad_alpha_term1 = vec![0.0; k_kernels];
        let mut r = vec![0.0; k_kernels];

        let lambda_0 = mu;
        grad_mu_term1 += 1.0 / lambda_0;
        for r_k in r.iter_mut() {
            *r_k += 1.0;
        }

        for i in 1..n {
            let delta_t = self.timestamps[i] - self.timestamps[i - 1];

            let mut excitation = 0.0;
            for k in 0..k_kernels {
                r[k] *= (-self.fixed_betas[k] * delta_t).exp();
                excitation += alphas[k] * r[k];
            }

            let lambda_t = mu + excitation;
            let inv_lambda = 1.0 / lambda_t;

            grad_mu_term1 += inv_lambda;
            for alpha_grad in grad_alpha_term1.iter_mut().zip(r.iter()) {
                *alpha_grad.0 += alpha_grad.1 * inv_lambda;
            }
            for r_k in r.iter_mut() {
                *r_k += 1.0;
            }
        }

        let t_max = self.timestamps.last().copied().unwrap_or(0.0);
        let t_min = self.timestamps.first().copied().unwrap_or(0.0);
        let total_duration = t_max - t_min;

        let grad_integral_mu = total_duration;
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
            grad_log
                .push(alphas[k] * (grad_actual[1 + k] - weighted_alpha_grad / self.fixed_betas[k]));
        }

        Ok(grad_log)
    }
}

pub fn fit_hawkes(
    timestamps: Vec<f64>,
    fixed_betas: Vec<f64>,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    let cost = HawkesLikelihood::new(timestamps, fixed_betas)?;

    // Initial guess:
    // mu = 1.0
    // alpha_k / beta_k share is initialized to about 0.2 / K, keeping the
    // total branching ratio strictly below 1.
    let mut init_param = vec![1.0f64.ln()];
    let target_branch_component = (0.2 / cost.fixed_betas.len() as f64).ln();
    for _ in &cost.fixed_betas {
        init_param.push(target_branch_component);
    }

    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001)?);
    let solver = LBFGS::new(linesearch, 7)
        .with_tolerance_grad(1e-4)?
        .with_tolerance_cost(1e-4)?;

    let res = Executor::new(cost.clone(), solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .run()?;

    // Convert back from log params
    let best_param_log = res.state.best_param.ok_or("No best parameter found")?;
    let (mu, alphas) = cost.convert_params(&best_param_log).unwrap();

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
    fn test_precomputed_integral_sums_match_reference() {
        let likelihood = HawkesLikelihood::new(vec![0.0, 1.0, 3.0], vec![2.0, 0.5]).unwrap();
        let expected = compute_integral_sums(&likelihood.timestamps, &likelihood.fixed_betas);

        assert_eq!(likelihood.integral_sums(), expected.as_slice());
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
}
