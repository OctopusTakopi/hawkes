//! Demonstrates a finite-support boxcar kernel implemented via `HawkesModel`.

use hawkes::{HawkesError, HawkesModel, HawkesResult};
use std::collections::VecDeque;

/// Univariate Hawkes state with `g(t) = alpha` for `0 <= t < support`.
#[derive(Debug)]
struct BoxcarHawkes {
    mu: f64,
    alpha: f64,
    support_us: u64,
    active_events: VecDeque<(u64, f64)>,
    current_excitation: f64,
    last_timestamp_us: Option<u64>,
}

impl BoxcarHawkes {
    fn new(mu: f64, alpha: f64, support_us: u64) -> Self {
        Self {
            mu,
            alpha,
            support_us,
            active_events: VecDeque::new(),
            current_excitation: 0.0,
            last_timestamp_us: None,
        }
    }

    fn validate_timestamp(&self, timestamp_us: u64) -> HawkesResult<()> {
        if let Some(previous_us) = self.last_timestamp_us
            && timestamp_us < previous_us
        {
            return Err(HawkesError::NonMonotonicTimestamp {
                previous_us,
                current_us: timestamp_us,
            });
        }
        Ok(())
    }

    fn is_active(&self, event_us: u64, timestamp_us: u64) -> bool {
        timestamp_us - event_us < self.support_us
    }
}

impl HawkesModel for BoxcarHawkes {
    fn update(&mut self, timestamp_us: u64, volume: Option<f64>) -> HawkesResult<f64> {
        self.validate_timestamp(timestamp_us)?;
        let volume = volume.unwrap_or(1.0);
        if !volume.is_finite() {
            return Err(HawkesError::NonFiniteParameter("volume", volume));
        }
        if volume < 0.0 {
            return Err(HawkesError::InvalidVolume(volume));
        }

        while self
            .active_events
            .front()
            .is_some_and(|&(event_us, _)| !self.is_active(event_us, timestamp_us))
        {
            let (_, expired_volume) = self.active_events.pop_front().unwrap();
            self.current_excitation -= self.alpha * expired_volume;
        }

        self.active_events.push_back((timestamp_us, volume));
        self.current_excitation += self.alpha * volume;
        self.last_timestamp_us = Some(timestamp_us);
        Ok(self.current_excitation)
    }

    fn evaluate(&self, timestamp_us: u64) -> HawkesResult<f64> {
        self.validate_timestamp(timestamp_us)?;
        Ok(self
            .active_events
            .iter()
            .filter(|&&(event_us, _)| self.is_active(event_us, timestamp_us))
            .map(|&(_, volume)| self.alpha * volume)
            .sum())
    }

    fn current_excitation(&self) -> f64 {
        self.current_excitation
    }

    fn intensity(&self) -> f64 {
        self.mu + self.current_excitation
    }
}

fn main() -> HawkesResult<()> {
    // Kernel mass is alpha * support = 0.2 * 2 = 0.4, hence stationary for unit marks.
    let mut model = BoxcarHawkes::new(0.5, 0.2, 2_000_000);
    model.update(0, None)?;
    model.update(1_000_000, None)?;

    println!("intensity after second event: {}", model.intensity());
    println!("excitation at 2.5 s: {}", model.evaluate(2_500_000)?);
    Ok(())
}
