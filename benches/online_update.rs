//! Throughput smoke benchmark for allocation-free online updates.

use hawkes::SumExpHawkes;
use std::hint::black_box;
use std::time::Instant;

#[cfg(debug_assertions)]
const EVENT_COUNT: u64 = 100_000;
#[cfg(not(debug_assertions))]
const EVENT_COUNT: u64 = 5_000_000;

fn main() {
    let mut model = SumExpHawkes::new(1.0, vec![10.0, 1.0, 0.1, 0.01], vec![100.0, 10.0, 1.0, 0.1])
        .expect("benchmark parameters are valid");

    let started = Instant::now();
    for event in 0..EVENT_COUNT {
        black_box(
            model
                .update(event * 1_000, None)
                .expect("timestamps are monotonic"),
        );
    }
    let elapsed = started.elapsed();
    let nanoseconds_per_event = elapsed.as_nanos() as f64 / EVENT_COUNT as f64;

    println!(
        "sum-exp online update: {EVENT_COUNT} events in {elapsed:.3?} ({nanoseconds_per_event:.2} ns/event)"
    );
}
