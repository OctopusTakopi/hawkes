use hawkes::HawkesExcitation;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Initialize the model
    // Mu (baseline) = 1.0, Alpha (jump) = 0.5, Beta (decay) = 2.0
    let mut model = HawkesExcitation::new(1.0, 0.5, 2.0)?;

    println!("Initialized Basic Hawkes Model: {:?}", model);
    println!("Initial Intensity: {:.4}", model.intensity());

    // 2. Simulate some events
    // Event at t=0ms with default volume (1.0)
    let e1 = model.update(0, None);
    println!(
        "t=0ms: Excitation={:.4}, Intensity={:.4}",
        e1,
        model.intensity()
    );

    // Check decay at t=100ms without update
    println!(
        "t=100ms (eval): Intensity={:.4}",
        model.mu + model.evaluate(100)
    );

    // Event at t=200ms
    let e2 = model.update(200, None);
    println!(
        "t=200ms: Excitation={:.4}, Intensity={:.4}",
        e2,
        model.intensity()
    );

    // 3. Serialize state
    let json = serde_json::to_string_pretty(&model)?;
    println!("\nSerialized State:\n{}", json);

    Ok(())
}
