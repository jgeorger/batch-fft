use clap::Parser;
use num_complex::Complex;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex32};
use std::time::Instant;

/// Batch 1D Complex-to-Complex FFT processor
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of FFTs in the batch
    #[arg(short, long)]
    batch: usize,

    /// FFT transform length
    #[arg(short, long)]
    length: usize,

    /// Number of threads to use
    #[arg(short, long)]
    threads: usize,
}

fn main() {
    let args = Args::parse();

    // Set the number of threads for rayon
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();

    println!("Batch 1D FFT Configuration:");
    println!("  Batch size: {}", args.batch);
    println!("  FFT length: {}", args.length);
    println!("  Threads: {}", args.threads);
    println!();

    // Initialize input data: batch of signals in a contiguous array
    // Total size: batch * length complex numbers
    let total_size = args.batch * args.length;
    let mut data: Vec<Complex32> = (0..total_size)
        .map(|i| {
            // Generate sample data (sine wave with varying frequencies)
            let t = (i % args.length) as f32 / args.length as f32;
            let batch_idx = i / args.length;
            let freq = 1.0 + batch_idx as f32;
            Complex::new((2.0 * std::f32::consts::PI * freq * t).cos(), 0.0)
        })
        .collect();

    println!("Initialized {} complex numbers ({} batches × {} length)",
             total_size, args.batch, args.length);

    // Perform batch FFT with timing
    let start = Instant::now();
    perform_batch_fft(&mut data, args.batch, args.length);
    let duration = start.elapsed();

    // Calculate performance metrics
    let time_ms = duration.as_secs_f64() * 1000.0;
    let flops = calculate_flops(args.batch, args.length);
    let gflops = flops / duration.as_secs_f64() / 1e9;

    // Output results
    println!();
    println!("Results:");
    println!("  Batch size: {}", args.batch);
    println!("  FFT length: {}", args.length);
    println!("  Threads: {}", args.threads);
    println!("  Execution time: {:.3} ms", time_ms);
    println!("  Total FLOPs: {:.2e}", flops);
    println!("  Performance: {:.3} GFLOPS", gflops);
}

/// Perform batch FFT processing using parallel execution
fn perform_batch_fft(data: &mut [Complex32], _batch: usize, length: usize) {
    // Create FFT planner (thread-safe)
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(length);

    // Process batches in parallel
    data.par_chunks_mut(length)
        .for_each(|signal| {
            // Each thread gets its own scratch buffer
            let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

            // Perform in-place FFT
            fft.process_with_scratch(signal, &mut scratch);
        });
}

/// Calculate the number of floating-point operations for batch FFT
/// For a complex FFT of length N: ~5*N*log2(N) FLOPs
/// For a batch of B FFTs: B * 5 * N * log2(N)
fn calculate_flops(batch: usize, length: usize) -> f64 {
    let n = length as f64;
    let b = batch as f64;
    b * 5.0 * n * n.log2()
}
