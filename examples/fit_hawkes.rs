use hawkes::SumExpHawkes;
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "data/BTCUSDT-aggTrades-2026-02-10.zip";
    if !Path::new(file_path).exists() {
        eprintln!(
            "Error: {} not found. Please run this example from the project root with the data file present.",
            file_path
        );
        return Ok(());
    }

    println!("Loading data from {}...", file_path);
    let file = std::fs::File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let file_in_zip = archive.by_index(0)?;

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file_in_zip);

    let mut timestamps = Vec::new();
    for result in rdr.records() {
        let record = result?;
        // Column 5 is timestamp in micros
        let t_micros: u64 = record[5].parse()?;
        timestamps.push(t_micros as f64 / 1_000_000.0);
    }

    // Sort timestamps (required for fitting)
    timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Limit to 200k events for faster demonstration
    if timestamps.len() > 200_000 {
        timestamps.truncate(200_000);
        println!("Truncated to 200,000 events for faster fitting.");
    }

    println!("Loaded {} events. Starting fitting...", timestamps.len());

    let fixed_betas = vec![10.0, 1.0, 0.1];

    let start = std::time::Instant::now();
    let model = SumExpHawkes::fit(timestamps, fixed_betas.clone())?;
    let duration = start.elapsed();

    println!("Fitting completed in {:.2?}", duration);
    println!("Fitted Model: {:?}", model);
    println!("Mu: {:.}", model.mu);
    for (i, alpha) in model.alphas.iter().enumerate() {
        println!("Alpha_{} (beta={}): {:.8}", i, fixed_betas[i], alpha);
    }

    let json = serde_json::to_string_pretty(&model)?;
    let output_path = "data/parameters.generated.json";
    std::fs::write(output_path, json)?;
    println!("Fitted parameters saved to {}", output_path);

    Ok(())
}
