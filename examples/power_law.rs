use hawkes::ApproxPowerLawHawkes;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Initialize Approximate Power Law Model
    // Fits a power law kernel g(t) = alpha / (delta + t)^beta using a sum of exponentials
    // Mu = 0.5
    // Alpha = 1.0 (scale)
    // Beta = 1.5 (power law exponent, typically > 1)
    // Delta = 0.01 (shift to avoid singularity at t=0)
    // K = 5 (number of exponential kernels to use for approximation)
    let mut model = ApproxPowerLawHawkes::new(0.5, 1.0, 1.5, 0.01, 5)?;

    // Alternatively, perform the approximation manually if you want to inspect scales
    // let manual_model = ApproxPowerLawHawkes::with_scales(...);

    println!("Initialized Power Law Approximation: {:?}", model);

    // Verify update matches evaluate
    let _e1 = model.update(0, None);
    let _e2 = model.update(500, None);

    // Predict future intensity
    let _future_intensity = model.evaluate(1000);
    // println!("Intensity at t=1000: {}", future_intensity);

    // Verify long-term decay
    for t in (1000..10000).step_by(1000) {
        model.evaluate(t);
    }

    let _intensity_at_1000 = model.evaluate(100000);
    // This returns just excitation
    // Accessing mu directly is not exposed on ApproxPowerLawHawkes wrapper by default unless I made it pub.

    // We want intensity at t=1000.
    // Note: the model's last timestamp is updated by `update`.
    // So `evaluate(1000)` will calculate decay from last event to 1000.

    // So to get intensity at t=1000:
    // let intensity_at_1000 = 0.5 + model.evaluate(1000);
    println!(
        "t=1000ms (eval): Intensity={:.4} (approx)",
        0.5 + model.evaluate(1000)
    );

    Ok(())
}
