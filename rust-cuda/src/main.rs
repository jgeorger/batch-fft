mod cuda_ffi;
mod cufft_ffi;

use clap::Parser;
use cuda_ffi::{CudaBuffer, CudaEvent};
use cufft_ffi::CufftPlan;
use num_complex::Complex32;
use std::f64::consts::PI;

#[derive(Parser, Debug)]
#[command(about = "Batch FFT processing with CUDA/cuFFT via Rust FFI")]
struct Args {
    #[arg(short, long, help = "Number of FFTs in the batch")]
    batch: usize,

    #[arg(short, long, help = "FFT transform length")]
    length: usize,
}

/// Generate input data with sine waves
/// Matches cuda/src/batch_fft.cu:37-48
fn generate_input_data(batch: usize, length: usize) -> Vec<Complex32> {
    let total_size = batch * length;
    let mut data = Vec::with_capacity(total_size);

    for i in 0..total_size {
        let t = ((i % length) as f64) / (length as f64);
        let batch_idx = i / length;
        let freq = 1.0 + (batch_idx as f64);

        // Real part: cos(2π * freq * t)
        // Imaginary part: 0.0
        let real = (2.0 * PI * freq * t).cos() as f32;
        data.push(Complex32::new(real, 0.0));
    }

    data
}

/// Calculate theoretical FLOPS for batch FFT
/// Matches cuda/src/batch_fft.cu:80-84
fn calculate_flops(batch: usize, length: usize) -> f64 {
    let n = length as f64;
    let b = batch as f64;
    b * 5.0 * n * n.log2()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Generate input data on host
    // Matches cuda/src/batch_fft.cu:94-97
    let total_size = args.batch * args.length;
    let h_data = generate_input_data(args.batch, args.length);

    // Allocate device memory
    // Matches cuda/src/batch_fft.cu:99-101
    let mut d_data = CudaBuffer::<Complex32>::alloc(total_size)?;

    // Copy data to device (NOT timed)
    // Matches cuda/src/batch_fft.cu:103-105
    d_data.copy_from_host(&h_data)?;

    // Create cuFFT plan
    let plan = CufftPlan::new_batch_1d(args.length, args.batch)?;

    // Warm-up run for accurate timing
    // Matches cuda/src/batch_fft.cu:107-108
    plan.execute_forward(d_data.as_ptr())?;

    // Re-copy data for actual timed run
    // Matches cuda/src/batch_fft.cu:110-112
    d_data.copy_from_host(&h_data)?;

    // Create CUDA events for timing
    // Matches cuda/src/batch_fft.cu:114-117
    let start = CudaEvent::create()?;
    let stop = CudaEvent::create()?;

    // Perform timed FFT
    // Matches cuda/src/batch_fft.cu:119-123
    start.record()?;
    plan.execute_forward(d_data.as_ptr())?;
    stop.record()?;
    stop.synchronize()?;

    // Calculate elapsed time
    // Matches cuda/src/batch_fft.cu:125-127
    let milliseconds = start.elapsed_time(&stop)?;

    // Calculate performance metrics
    // Matches cuda/src/batch_fft.cu:129-131
    let flops = calculate_flops(args.batch, args.length);
    let gflops = flops / ((milliseconds / 1000.0) as f64) / 1e9;

    // Output results as CSV
    // Matches cuda/src/batch_fft.cu:133-137
    println!("batch,fft_length,time_ms,gflops");
    println!(
        "{},{},{:.3},{:.0}",
        args.batch, args.length, milliseconds, gflops
    );

    Ok(())
}
