//! Reproducible chronological audit on the 2019 Ridgecrest earthquake sequence.

use hawkes::diagnostics::{score_hawkes, score_poisson};
use hawkes::{Baseline, SumExpHawkes};
use serde::Deserialize;
use std::error::Error;
use std::time::{Duration, Instant};

const DATA: &[u8] = include_bytes!("../data/ridgecrest_2019_m2_5.csv");
const SECONDS_PER_DAY: f64 = 86_400.0;
const MEAN_COMPENSATOR_TOLERANCE: f64 = 0.10;
const MAX_ABS_LAG1_AUTOCORRELATION: f64 = 0.10;
const BRANCHING_BOUNDARY_TOLERANCE: f64 = 1.0e-6;

#[derive(Deserialize)]
struct Event {
    seconds_since_2019_07_04_utc: f64,
}

fn load_timestamps() -> Result<Vec<f64>, csv::Error> {
    csv::Reader::from_reader(DATA)
        .deserialize::<Event>()
        .map(|event| event.map(|event| event.seconds_since_2019_07_04_utc))
        .collect()
}

struct AuditResult {
    model: SumExpHawkes,
    total_events: usize,
    train_events: usize,
    test_events: usize,
    window_days: f64,
    fit_elapsed: Duration,
    hawkes_log_likelihood: f64,
    poisson_log_likelihood: f64,
    lift_per_event: f64,
    mean_compensator: f64,
    ks_statistic: f64,
    ks_critical_value: f64,
    lag1_autocorrelation: f64,
    has_interior_branching_ratio: bool,
    passes_calibration: bool,
    is_adequate: bool,
}

fn run_audit() -> Result<AuditResult, Box<dyn Error>> {
    let timestamps = load_timestamps()?;
    let split_index = timestamps.len() * 4 / 5;
    let origin = timestamps[0];
    let relative_timestamps = timestamps
        .iter()
        .map(|timestamp| timestamp - origin)
        .collect::<Vec<_>>();
    let train_timestamps = relative_timestamps[..split_index].to_vec();

    // Five-minute, one-hour, and one-day decay scales, fixed before fitting.
    let fixed_betas = vec![1.0 / 300.0, 1.0 / 3_600.0, 1.0 / SECONDS_PER_DAY];
    let fit_started = Instant::now();
    let model = SumExpHawkes::fit(train_timestamps, fixed_betas)?;
    let fit_elapsed = fit_started.elapsed();

    let marks = vec![1.0; relative_timestamps.len()];
    let hawkes_score = score_hawkes(&model, &relative_timestamps, &marks, split_index)?;
    let train_duration = relative_timestamps[split_index - 1] - relative_timestamps[0];
    let poisson_rate = (split_index - 1) as f64 / train_duration;
    let poisson_score = score_poisson(
        &Baseline::constant(poisson_rate)?,
        &relative_timestamps,
        split_index,
    )?;
    let test_events = hawkes_score.event_count();
    let hawkes_log_likelihood = hawkes_score.log_likelihood();
    let poisson_log_likelihood = poisson_score.log_likelihood();
    let lift_per_event = (hawkes_log_likelihood - poisson_log_likelihood) / test_events as f64;
    let diagnostics = hawkes_score.time_rescaling();
    let has_interior_branching_ratio = model.branching_ratio() < 1.0 - BRANCHING_BOUNDARY_TOLERANCE;
    let passes_calibration = (diagnostics.mean_compensator() - 1.0).abs()
        <= MEAN_COMPENSATOR_TOLERANCE
        && diagnostics.passes_ks_5pct()
        && diagnostics.lag1_autocorrelation().abs() <= MAX_ABS_LAG1_AUTOCORRELATION;
    let is_adequate = lift_per_event > 0.0 && has_interior_branching_ratio && passes_calibration;

    Ok(AuditResult {
        model,
        total_events: relative_timestamps.len(),
        train_events: split_index,
        test_events,
        window_days: (relative_timestamps[relative_timestamps.len() - 1] - relative_timestamps[0])
            / SECONDS_PER_DAY,
        fit_elapsed,
        hawkes_log_likelihood,
        poisson_log_likelihood,
        lift_per_event,
        mean_compensator: diagnostics.mean_compensator(),
        ks_statistic: diagnostics.ks_statistic(),
        ks_critical_value: diagnostics.ks_critical_value_5pct(),
        lag1_autocorrelation: diagnostics.lag1_autocorrelation(),
        has_interior_branching_ratio,
        passes_calibration,
        is_adequate,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let audit = run_audit()?;

    println!("Ridgecrest 2019 temporal holdout audit");
    println!(
        "events: {} total, {} train / {} test; window: {:.2} days",
        audit.total_events, audit.train_events, audit.test_events, audit.window_days
    );
    println!("fit time: {:.3?}", audit.fit_elapsed);
    println!(
        "mu: {:.8}; branching ratio: {:.8}",
        audit.model.mu(),
        audit.model.branching_ratio()
    );
    println!("alphas: {:?}", audit.model.alphas());
    println!("betas:  {:?}", audit.model.betas());
    println!(
        "holdout log-likelihood: Hawkes {:.3}, Poisson {:.3}",
        audit.hawkes_log_likelihood, audit.poisson_log_likelihood
    );
    println!("predictive lift: {:.6} nats/event", audit.lift_per_event);
    println!(
        "time-rescaling: mean increment {:.6} (target 1), KS D {:.6} (5% critical {:.6}), lag-1 r {:+.6}",
        audit.mean_compensator,
        audit.ks_statistic,
        audit.ks_critical_value,
        audit.lag1_autocorrelation
    );
    println!(
        "audit gates: branching interior {}, residual calibration {}",
        audit.has_interior_branching_ratio, audit.passes_calibration
    );
    println!(
        "model adequacy: {}",
        if audit.is_adequate { "PASS" } else { "REJECT" }
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridgecrest_fit_is_stationary_and_beats_poisson_holdout() {
        let audit = run_audit().unwrap();

        assert_eq!(audit.total_events, 2_062);
        assert!(audit.model.branching_ratio() < 1.0);
        assert!(audit.hawkes_log_likelihood.is_finite());
        assert!(audit.lift_per_event > 0.0);
    }

    #[test]
    fn ridgecrest_audit_rejects_the_miscalibrated_specification() {
        let audit = run_audit().unwrap();

        assert!(!audit.has_interior_branching_ratio);
        assert!(!audit.passes_calibration);
        assert!(!audit.is_adequate);
    }
}
