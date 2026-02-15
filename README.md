# Hawkes Process for Rust

A high-performance library for modeling and fitting Hawkes processes (self-exciting point processes).

## Features
- **Exponential Kernel**: Fast $O(1)$ updates per event.
- **Sum-of-Exponentials**: Flexible $O(K)$ multi-scale modeling.
- **Approximate Power-Law**: Using log-spaced sum-of-exponentials.
- **Maximum Likelihood Estimation (MLE)**: Fit model parameters to event data using L-BFGS.

## Usage

Add this to your `Cargo.toml`:
```toml
[dependencies]
hawkes = "0.1"
```

### Fitting a Model
```rust
use hawkes::SumExpHawkes;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Event timestamps in seconds
    let timestamps = vec![1.2, 1.4, 2.5, 2.6, 2.7, 4.0];
    
    // Fixed decay rates (hyperparameters)
    let betas = vec![1.0, 10.0];
    
    // Fit mu and alphas
    let model = SumExpHawkes::fit(timestamps, betas)?;
    
    println!("Fitted Model: {:?}", model);
    Ok(())
}
```

### Real-time Intensity Tracking
```rust
use hawkes::{HawkesExcitation, HawkesModel};

fn main() {
    let mut model = HawkesExcitation::new(0.1, 0.5, 1.0).unwrap();
    
    // Process events as they arrive (timestamp in milliseconds)
    let intensity = model.update(1000, None);
    println!("Intensity after event: {}", model.intensity());
}
```

## License
MIT
