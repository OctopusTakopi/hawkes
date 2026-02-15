//! Hawkes Process parameter fitting using Maximum Likelihood Estimation (MLE).
//!
//! This module provides the infrastructure to fit Hawkes Process parameters (mu, alphas)
//! to a sequence of timestamps using the L-BFGS optimization algorithm.

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
}

impl HawkesLikelihood {
    pub fn new(timestamps: Vec<f64>, fixed_betas: Vec<f64>) -> Self {
        Self {
            timestamps,
            fixed_betas,
        }
    }

    /// Computes NegLogLikelihood and Gradient.
    /// Returns (nll, grad)
    /// Params: [log_mu, log_alpha_0, log_alpha_1, ...]
    /// Internal parameters are LOG of actual parameters to enforce positivity.
    pub fn convert_params(&self, p: &[f64]) -> Option<(f64, Vec<f64>)> {
        if p.len() != 1 + self.fixed_betas.len() {
            return None;
        }
        // Exponentiate to get actual positive parameters
        // Guard against overflow: if p > 50.0, exp(p) is huge.
        // Bounds check: if any param > 50.0, return None (Cost = Inf)
        if p.iter().any(|&x| x > 50.0) {
            return None;
        }

        let mu = p[0].exp();
        let alphas: Vec<f64> = p[1..].iter().map(|x| x.exp()).collect();

        // No more bounds check needed as exp() > 0
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
            .timestamps
            .par_iter()
            .fold(
                || vec![0.0; k_kernels],
                |mut acc, &t| {
                    for (k, acc_k) in acc.iter_mut().enumerate() {
                        *acc_k += 1.0 - (-self.fixed_betas[k] * (t_max - t)).exp();
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0; k_kernels],
                |mut a, b| {
                    for (a_k, b_k) in a.iter_mut().zip(b.iter()) {
                        *a_k += *b_k;
                    }
                    a
                },
            )
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
            None => return Ok(vec![0.0; param.len()]),
        };

        // ... Standard gradient calculation w.r.t actual parameters mu, alphas ...
        let n = self.timestamps.len();
        let k_kernels = self.fixed_betas.len();

        // Debug params
        // println!("Gradient eval at params: {:?}", param);
        // println!("Gradient eval at actual: mu={}, alphas={:?}", mu, alphas);

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

        let term_sums: Vec<f64> = self
            .timestamps
            .par_iter()
            .fold(
                || vec![0.0; k_kernels],
                |mut acc, &t| {
                    for (k, acc_k) in acc.iter_mut().enumerate() {
                        *acc_k += 1.0 - (-self.fixed_betas[k] * (t_max - t)).exp();
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0; k_kernels],
                |mut a, b| {
                    for (a_k, b_k) in a.iter_mut().zip(b.iter()) {
                        *a_k += *b_k;
                    }
                    a
                },
            );

        for k in 0..k_kernels {
            grad_integral_alpha[k] = term_sums[k] / self.fixed_betas[k];
        }

        // Combine for NegLogLikelihood Gradient w.r.t actual params
        let mut grad_actual = Vec::with_capacity(param.len());
        grad_actual.push(-(grad_mu_term1 - grad_integral_mu));
        for k in 0..k_kernels {
            grad_actual.push(-(grad_alpha_term1[k] - grad_integral_alpha[k]));
        }

        // Apply Chain Rule for Log-Parameters
        // param = log(p_actual) => p_actual = exp(param)
        // d(Cost)/d(param) = d(Cost)/d(p_actual) * d(p_actual)/d(param)
        //                  = grad_actual * exp(param)
        //                  = grad_actual * p_actual
        let mut grad_log = Vec::with_capacity(param.len());

        // Mu
        grad_log.push(grad_actual[0] * mu);
        // Alphas
        for k in 0..k_kernels {
            grad_log.push(grad_actual[1 + k] * alphas[k]);
        }

        // Debug gradient
        // println!("Computed Gradient: {:?}", grad_log);

        Ok(grad_log)
    }
}

pub fn fit_hawkes(
    timestamps: Vec<f64>,
    fixed_betas: Vec<f64>,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    let cost = HawkesLikelihood::new(timestamps, fixed_betas);

    // Initial guess: Use LOG parameters.
    // Try mu=1.0, alpha=0.2*beta
    let mut init_param = vec![1.0f64.ln()];
    for &beta in &cost.fixed_betas {
        init_param.push((beta * 0.2).ln());
    }

    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001)?);
    let solver = LBFGS::new(linesearch, 7)
        .with_tolerance_grad(1e-4)?
        .with_tolerance_cost(1e-4)?;

    let res = Executor::new(cost.clone(), solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .run()?;

    // println!("Iterations: {}", res.state.get_iter());
    // println!("Final Cost: {}", res.state.get_cost());
    // println!("Termination: {:?}", res.state.get_termination_reason());

    // Convert back from log params
    let best_param_log = res.state.best_param.ok_or("No best parameter found")?;
    let (mu, alphas) = cost.convert_params(&best_param_log).unwrap();

    // println!(
    //    "Fitted Model: {:?}",
    //    SumExpHawkes::new(mu, alphas.clone(), cost.fixed_betas.clone())?
    // );

    Ok((mu, alphas))
}
