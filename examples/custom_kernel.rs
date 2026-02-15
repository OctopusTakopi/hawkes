use hawkes::{HawkesModel, HawkesResult};

/// A custom Kernel implementing the HawkesModel trait.
/// Example: A simple linear decay kernel g(t) = max(0, alpha - beta * t)
/// Note: This is just for demonstration; linear kernels are rarely used due to negative possibilities,
/// but it shows how to extend the library.
#[derive(Debug)]
struct LinearDecayHawkes {
    mu: f64,
    alpha: f64,
    beta: f64,
    last_timestamp: u64,
    current_val: f64,
}

impl LinearDecayHawkes {
    fn new(mu: f64, alpha: f64, beta: f64) -> Self {
        Self {
            mu,
            alpha,
            beta,
            last_timestamp: 0,
            current_val: 0.0,
        }
    }
}

impl HawkesModel for LinearDecayHawkes {
    fn update(&mut self, timestamp_ms: u64, _volume: Option<f64>) -> f64 {
        let dt = (timestamp_ms - self.last_timestamp) as f64 / 1000.0;

        // Linear decay: x(t) = x(t-1) - beta * dt
        let decayed = (self.current_val - self.beta * dt).max(0.0);

        // Jump
        self.current_val = decayed + self.alpha;
        self.last_timestamp = timestamp_ms;

        self.current_val
    }

    fn evaluate(&self, timestamp_ms: u64) -> f64 {
        let dt = (timestamp_ms - self.last_timestamp) as f64 / 1000.0;
        (self.current_val - self.beta * dt).max(0.0)
    }

    fn current_excitation(&self) -> f64 {
        self.current_val
    }

    fn intensity(&self) -> f64 {
        self.mu + self.current_val
    }
}

fn main() -> HawkesResult<()> {
    let mut custom_model = LinearDecayHawkes::new(1.0, 2.0, 0.5);
    println!("Custom Kernel Initialized: {:?}", custom_model);

    custom_model.update(0, None);
    println!("t=0: Intensity={}", custom_model.intensity());

    custom_model.update(1000, None); // dt=1.0s, decay = 0.5
    println!(
        "t=1000 (after 1s decay): Intensity={}",
        custom_model.intensity()
    );

    Ok(())
}
