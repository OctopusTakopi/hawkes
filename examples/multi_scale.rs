use hawkes::SumExpHawkes;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Initialize Sum of Exponentials Model
    // Commonly used to capture short-term and long-term effects
    // Mu = 1.0, Alphas = [0.5, 0.2], Betas = [10.0, 1.0]
    // Kernel 1: fast decay (beta=10), strong impact (alpha=0.5)
    // Kernel 2: slow decay (beta=1), weaker impact (alpha=0.2)
    let mut model = SumExpHawkes::new(1.0, vec![0.5, 0.2], vec![10.0, 1.0])?;

    println!("Initialized Multi-Scale Hawkes Model: {:?}", model);
    println!("Initial Intensity: {:.4}", model.intensity());

    // 2. Simulate rapid succession of events
    let timestamps = vec![0, 10_000, 20_000, 50_000, 100_000];

    for t in timestamps {
        let e = model.update(t, None)?;
        println!(
            "t={}us: Excitation={:.4}, Intensity={:.4}",
            t,
            e,
            model.intensity()
        );
    }

    Ok(())
}
