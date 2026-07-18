//! Constant and periodic baseline intensities.

use crate::{HawkesError, HawkesResult};
use serde::{Deserialize, Deserializer, Serialize};

/// A positive, piecewise-constant periodic shape with mean multiplier one.
///
/// The bins have equal width. A shape changes the timing of baseline activity
/// without changing its mean rate over a complete period.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PeriodicShape {
    period: f64,
    multipliers: Vec<f64>,
}

impl PeriodicShape {
    /// Creates a normalized periodic shape from positive bin multipliers.
    pub fn new(period: f64, multipliers: Vec<f64>) -> HawkesResult<Self> {
        validate_period(period)?;
        if multipliers.is_empty() {
            return Err(HawkesError::InvalidBinCount(0));
        }
        validate_bin_width(period, multipliers.len())?;
        if multipliers
            .iter()
            .any(|&multiplier| !multiplier.is_finite() || multiplier <= 0.0)
        {
            return Err(HawkesError::InvalidSeasonalMultiplier);
        }

        let mean = multipliers.iter().sum::<f64>() / multipliers.len() as f64;
        if !mean.is_finite() || mean <= 0.0 {
            return Err(HawkesError::InvalidSeasonalMultiplier);
        }

        Ok(Self {
            period,
            multipliers: multipliers
                .into_iter()
                .map(|multiplier| multiplier / mean)
                .collect(),
        })
    }

    /// Estimates a periodic shape from a strictly increasing training window.
    ///
    /// The first event is treated as conditioning history. Each bin rate is
    /// shrunk toward the global training rate using `smoothing_seconds` of
    /// pseudo-exposure, then the resulting shape is normalized to mean one.
    /// This is a marginal two-stage estimator: Hawkes offspring are counted as
    /// baseline events and can leak excitation clustering into the fitted shape.
    pub fn fit(
        timestamps: &[f64],
        period: f64,
        bin_count: usize,
        smoothing_seconds: f64,
    ) -> HawkesResult<Self> {
        validate_period(period)?;
        if bin_count == 0 {
            return Err(HawkesError::InvalidBinCount(bin_count));
        }
        validate_bin_width(period, bin_count)?;
        if !smoothing_seconds.is_finite() || smoothing_seconds <= 0.0 {
            return Err(HawkesError::InvalidSmoothing(smoothing_seconds));
        }
        validate_timestamps(timestamps)?;

        let start = timestamps[0];
        let end = timestamps[timestamps.len() - 1];
        let duration = end - start;
        let global_rate = (timestamps.len() - 1) as f64 / duration;
        let mut exposures = vec![0.0; bin_count];
        add_periodic_exposure(&mut exposures, period, start, end);

        let bin_width = period / bin_count as f64;
        let mut counts = vec![0_usize; bin_count];
        for &timestamp in &timestamps[1..] {
            counts[bin_index(timestamp, period, bin_width, bin_count)] += 1;
        }

        let multipliers = counts
            .into_iter()
            .zip(exposures)
            .map(|(count, exposure)| {
                let rate = (count as f64 + global_rate * smoothing_seconds)
                    / (exposure + smoothing_seconds);
                rate / global_rate
            })
            .collect();

        Self::new(period, multipliers)
    }

    /// Returns the period in the same time units as fitted timestamps.
    pub fn period(&self) -> f64 {
        self.period
    }

    /// Returns the equal-width bin multipliers.
    pub fn multipliers(&self) -> &[f64] {
        &self.multipliers
    }

    /// Returns the multiplier active at `timestamp`.
    pub fn multiplier(&self, timestamp: f64) -> HawkesResult<f64> {
        validate_timestamp(timestamp)?;
        let bin_width = self.period / self.multipliers.len() as f64;
        Ok(self.multipliers[bin_index(timestamp, self.period, bin_width, self.multipliers.len())])
    }

    /// Integrates the multiplier over `[start, end]`.
    pub fn integrated_multiplier(&self, start: f64, end: f64) -> HawkesResult<f64> {
        validate_interval(start, end)?;
        let integrated = integrate_periodic(&self.multipliers, self.period, start, end);
        if !integrated.is_finite() {
            return Err(HawkesError::NonFiniteParameter(
                "integrated_multiplier",
                integrated,
            ));
        }
        Ok(integrated)
    }
}

impl<'de> Deserialize<'de> for PeriodicShape {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            period: f64,
            multipliers: Vec<f64>,
        }

        let helper = Helper::deserialize(deserializer)?;
        Self::new(helper.period, helper.multipliers).map_err(serde::de::Error::custom)
    }
}

/// A validated constant or periodic baseline intensity.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Baseline {
    kind: BaselineKind,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
enum BaselineKind {
    Constant(f64),
    Periodic {
        mean_rate: f64,
        shape: PeriodicShape,
    },
}

impl Baseline {
    /// Creates a constant baseline.
    pub fn constant(rate: f64) -> HawkesResult<Self> {
        validate_rate(rate)?;
        Ok(Self {
            kind: BaselineKind::Constant(rate),
        })
    }

    /// Creates a periodic baseline with the supplied mean rate.
    pub fn periodic(mean_rate: f64, shape: PeriodicShape) -> HawkesResult<Self> {
        validate_rate(mean_rate)?;
        Ok(Self {
            kind: BaselineKind::Periodic { mean_rate, shape },
        })
    }

    /// Returns the baseline intensity at `timestamp`.
    pub fn intensity(&self, timestamp: f64) -> HawkesResult<f64> {
        validate_timestamp(timestamp)?;
        let intensity = match &self.kind {
            BaselineKind::Constant(rate) => *rate,
            BaselineKind::Periodic { mean_rate, shape } => {
                *mean_rate * shape.multiplier(timestamp)?
            }
        };
        if !intensity.is_finite() || intensity <= 0.0 {
            return Err(HawkesError::InvalidBaselineRate(intensity));
        }
        Ok(intensity)
    }

    /// Integrates the baseline intensity over `[start, end]`.
    pub fn integrated_intensity(&self, start: f64, end: f64) -> HawkesResult<f64> {
        validate_interval(start, end)?;
        let integrated = match &self.kind {
            BaselineKind::Constant(rate) => *rate * (end - start),
            BaselineKind::Periodic { mean_rate, shape } => {
                *mean_rate * shape.integrated_multiplier(start, end)?
            }
        };
        if !integrated.is_finite() || integrated < 0.0 {
            return Err(HawkesError::NonFiniteParameter(
                "integrated_baseline",
                integrated,
            ));
        }
        Ok(integrated)
    }
}

fn validate_period(period: f64) -> HawkesResult<()> {
    if !period.is_finite() || period <= 0.0 {
        return Err(HawkesError::InvalidPeriod(period));
    }
    Ok(())
}

fn validate_rate(rate: f64) -> HawkesResult<()> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err(HawkesError::InvalidBaselineRate(rate));
    }
    Ok(())
}

fn validate_bin_width(period: f64, bin_count: usize) -> HawkesResult<()> {
    let bin_width = period / bin_count as f64;
    if !bin_width.is_finite() || bin_width <= 0.0 {
        return Err(HawkesError::InvalidBinCount(bin_count));
    }
    Ok(())
}

fn validate_timestamp(timestamp: f64) -> HawkesResult<()> {
    if !timestamp.is_finite() {
        return Err(HawkesError::NonFiniteTimestamp(timestamp));
    }
    Ok(())
}

fn validate_interval(start: f64, end: f64) -> HawkesResult<()> {
    validate_timestamp(start)?;
    validate_timestamp(end)?;
    if end < start {
        return Err(HawkesError::InvalidTimeInterval { start, end });
    }
    Ok(())
}

fn validate_timestamps(timestamps: &[f64]) -> HawkesResult<()> {
    if timestamps.len() < 2 {
        return Err(HawkesError::InvalidObservationWindow);
    }
    for &timestamp in timestamps {
        validate_timestamp(timestamp)?;
    }
    if timestamps.windows(2).any(|window| window[1] < window[0]) {
        return Err(HawkesError::UnsortedTimestamps);
    }
    if let Some(timestamp) = timestamps
        .windows(2)
        .find(|window| window[1] == window[0])
        .map(|window| window[0])
    {
        return Err(HawkesError::DuplicateTimestamp(timestamp));
    }
    if timestamps[timestamps.len() - 1] <= timestamps[0] {
        return Err(HawkesError::InvalidObservationWindow);
    }
    Ok(())
}

fn bin_index(timestamp: f64, period: f64, bin_width: f64, bin_count: usize) -> usize {
    ((timestamp.rem_euclid(period) / bin_width).floor() as usize).min(bin_count - 1)
}

fn add_periodic_exposure(exposures: &mut [f64], period: f64, start: f64, end: f64) {
    let duration = end - start;
    let bin_width = period / exposures.len() as f64;
    let full_periods = (duration / period).floor();
    for exposure in exposures.iter_mut() {
        *exposure = full_periods * bin_width;
    }

    let mut remaining = duration - full_periods * period;
    let mut phase = start.rem_euclid(period);
    while remaining > 0.0 {
        let index = ((phase / bin_width).floor() as usize).min(exposures.len() - 1);
        let boundary = ((index + 1) as f64 * bin_width).min(period);
        let width = (boundary - phase).min(remaining);
        if width <= 0.0 {
            phase = 0.0;
            continue;
        }
        exposures[index] += width;
        remaining -= width;
        phase += width;
        if phase >= period {
            phase = 0.0;
        }
    }
}

fn integrate_periodic(values: &[f64], period: f64, start: f64, end: f64) -> f64 {
    let duration = end - start;
    if duration == 0.0 {
        return 0.0;
    }

    let bin_width = period / values.len() as f64;
    let period_integral = values.iter().sum::<f64>() * bin_width;
    let full_periods = (duration / period).floor();
    let mut integral = full_periods * period_integral;
    let mut remaining = duration - full_periods * period;
    let mut phase = start.rem_euclid(period);

    while remaining > 0.0 {
        let index = ((phase / bin_width).floor() as usize).min(values.len() - 1);
        let boundary = ((index + 1) as f64 * bin_width).min(period);
        let width = (boundary - phase).min(remaining);
        if width <= 0.0 {
            phase = 0.0;
            continue;
        }
        integral += values[index] * width;
        remaining -= width;
        phase += width;
        if phase >= period {
            phase = 0.0;
        }
    }

    integral
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_is_normalized_and_integrates_across_boundaries() {
        let shape = PeriodicShape::new(4.0, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert!((shape.multipliers().iter().sum::<f64>() - 4.0).abs() < 1.0e-12);
        assert!((shape.integrated_multiplier(0.0, 4.0).unwrap() - 4.0).abs() < 1.0e-12);
        assert!((shape.integrated_multiplier(3.5, 4.5).unwrap() - 1.0).abs() < 1.0e-12);
        assert!((shape.integrated_multiplier(-0.5, 0.5).unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn fitted_shape_uses_training_exposure() {
        let timestamps = vec![0.0, 0.5, 1.5, 2.5, 3.5, 4.0];
        let shape = PeriodicShape::fit(&timestamps, 4.0, 4, 1.0).unwrap();

        assert_eq!(shape.multipliers().len(), 4);
        assert!(shape.multipliers().iter().all(|value| *value > 0.0));
        assert!((shape.integrated_multiplier(0.0, 4.0).unwrap() - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn fitted_shape_rejects_duplicate_timestamps() {
        assert_eq!(
            PeriodicShape::fit(&[0.0, 1.0, 1.0, 2.0], 2.0, 2, 1.0).unwrap_err(),
            HawkesError::DuplicateTimestamp(1.0)
        );
    }

    #[test]
    fn rejects_invalid_shapes() {
        assert!(matches!(
            PeriodicShape::new(0.0, vec![1.0]),
            Err(HawkesError::InvalidPeriod(0.0))
        ));
        assert!(matches!(
            PeriodicShape::new(1.0, vec![]),
            Err(HawkesError::InvalidBinCount(0))
        ));
        assert!(matches!(
            PeriodicShape::new(1.0, vec![0.0]),
            Err(HawkesError::InvalidSeasonalMultiplier)
        ));
    }
}
