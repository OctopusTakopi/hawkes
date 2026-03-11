use hawkes::{ApproxPowerLawHawkes, HawkesExcitation, SumExpHawkes};
use serde::{Deserialize, Serialize};

/// Represents a single trade event from the Binance @trade stream.
#[derive(Debug, Deserialize)]
pub struct BinanceTrade {
    #[serde(rename = "T")]
    pub trade_time: u64, // Trade timestamp in microseconds
    #[serde(rename = "q")]
    pub quantity: String, // Trade volume as a string
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Standard Exponential Hawkes (Single timescale)
    // Assuming mu=1.0 for standard example
    let mut std_hawkes = HawkesExcitation::new(1.0, 0.5, 2.0)?;

    // 2. Sum of Exponentials (Multi-timescale)
    // Load parameters from JSON
    #[derive(Serialize, Deserialize)]
    struct HawkesParams {
        mu: f64,
        alphas: Vec<f64>,
        betas: Vec<f64>,
    }

    let params_path = "data/parameters.json";
    let file_path = "data/BTCUSDT-aggTrades-2026-02-10.zip";

    // Define fixed timescales (betas)
    let fixed_betas = vec![10.0, 1.0, 0.1];

    let mut sum_hawkes = if std::path::Path::new(params_path).exists() {
        println!("Loading parameters from {}...", params_path);
        let file = std::fs::File::open(params_path)?;
        let params: HawkesParams = serde_json::from_reader(file)?;
        SumExpHawkes::new(params.mu, params.alphas, params.betas)?
    } else {
        println!("{} not found. Fitting model from scratch...", params_path);
        if !std::path::Path::new(file_path).exists() {
            return Err(format!("Data file {} not found for fitting.", file_path).into());
        }

        println!("Reading trades from {} for fitting...", file_path);
        let file = std::fs::File::open(file_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        let file_in_zip = archive.by_index(0)?;

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(file_in_zip);

        let mut timestamps: Vec<f64> = Vec::new();
        for result in rdr.records() {
            let record = result?;
            // Column 5 is timestamp in microseconds
            let t_micros: u64 = record[5].parse()?;
            timestamps.push(t_micros as f64 / 1_000_000.0);
        }
        timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("Loaded {} events. Starting fitting...", timestamps.len());

        let fitted_model = SumExpHawkes::fit(timestamps, fixed_betas.clone())?;

        println!("Fitting complete. Saving parameters to {}...", params_path);
        let params_to_save = HawkesParams {
            mu: fitted_model.mu,
            alphas: fitted_model.alphas.clone(),
            betas: fitted_model.betas.clone(),
        };
        let file = std::fs::File::create(params_path)?;
        serde_json::to_writer_pretty(file, &params_to_save)?;

        fitted_model
    };

    println!("Model Ready: {:?}", sum_hawkes);

    // 3. Approximate Power Law (Long memory)
    // Using 5 exponentials to approximate the (delta + t)^-1.5 curve
    let mut power_hawkes = ApproxPowerLawHawkes::new(1.0, 1.0, 1.5, 0.01, 5)?;

    // Open the CSV file
    // Open the CSV file (inside zip)
    let file_path = "data/BTCUSDT-aggTrades-2026-02-10.zip";
    println!("Reading trades from {}...", file_path);
    let file = std::fs::File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let file_in_zip = archive.by_index(0)?;

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file_in_zip);

    let mut count = 0;
    let start_time = std::time::Instant::now();

    for result in rdr.records() {
        let record = result?;
        // CSV Format: [id, price, qty, ..., timestamp_us, ...]
        // We verified timestamp is at index 5 (microseconds in the file we looked at earlier?
        // Wait, the head output showed: 3855286221,70138.00000000,0.00186000,5933596729,5933596729,1770681600109055,True,True
        // The offline_fit.py used column 5 and divided by 1,000,000.0, so it treated it as microseconds.
        // Let's use the same logic here.

        let timestamp_us: u64 = record[5].parse()?;
        let quantity: f64 = record[2].parse()?; // Index 2 is quantity based on head output

        // Update models
        let _e1 = std_hawkes.update(timestamp_us, Some(quantity))?;
        let e2 = sum_hawkes.update(timestamp_us, Some(quantity))?;
        let _e3 = power_hawkes.update(timestamp_us, Some(quantity))?;

        count += 1;
        if count % 100_000 == 0 {
            println!(
                "Processed {} trades. Current Excitation (SumExp): {:.4}",
                count, e2
            );
        }
    }

    println!(
        "Finished processing {} trades in {:.2?}",
        count,
        start_time.elapsed()
    );

    // Print final state
    println!(
        "\nFinal State:\n{:<15} | {:<12.4} | {:<12.4} | {:<12.4}",
        "Timestamp",
        std_hawkes.intensity(),
        sum_hawkes.intensity(),
        power_hawkes.intensity()
    );

    let saved_state = serde_json::to_string(&std_hawkes).unwrap();
    println!("Serialized Standard Hawkes: {}", saved_state);

    Ok(())
}
