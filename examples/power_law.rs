//! Compares the approximate kernel with its shifted power-law target.

use hawkes::ApproxPowerLawHawkes;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mu = 0.5;
    let alpha = 0.04;
    let exponent = 1.5;
    let delta = 0.01;
    let mut model = ApproxPowerLawHawkes::new(mu, alpha, exponent, delta, 100)?;

    model.update(0, None)?;
    println!("branching ratio: {:.6}", model.branching_ratio());
    println!("time (s) | approximation | target | relative error");

    for timestamp_us in [0, 1_000, 10_000, 100_000, 1_000_000] {
        let time = timestamp_us as f64 / 1_000_000.0;
        let approximation = model.evaluate(timestamp_us)?;
        let target = alpha / (delta + time).powf(exponent);
        let relative_error = (approximation - target).abs() / target;
        println!("{time:8.3} | {approximation:13.6} | {target:13.6} | {relative_error:13.6}");
    }

    Ok(())
}
