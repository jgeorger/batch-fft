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

    // Perform batch FFT with timing
    let start = Instant::now();
    perform_batch_fft(&mut data, args.batch, args.length);
    let duration = start.elapsed();

    // Calculate performance metrics
    let time_ms = duration.as_secs_f64() * 1000.0;
    let flops = calculate_flops(args.batch, args.length);
    let gflops = flops / duration.as_secs_f64() / 1e9;

    // Output results as CSV
    println!("batch,fft_length,threads,time_ms,gflops");
    println!("{},{},{},{:.3},{:.0}", args.batch, args.length, args.threads, time_ms, gflops);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_dc_signal() {
        // Test that a DC signal (constant value) has all energy in the DC bin (index 0)
        let length = 64;
        let dc_value = 5.0;
        let mut data: Vec<Complex32> = vec![Complex::new(dc_value, 0.0); length];

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(length);
        fft.process(&mut data);

        // DC bin should have value = length * dc_value
        let dc_magnitude = data[0].norm();
        let expected_dc = (length as f32) * dc_value;
        assert!((dc_magnitude - expected_dc).abs() < 1e-4,
                "DC bin magnitude {} should be close to {}", dc_magnitude, expected_dc);

        // All other bins should be near zero
        for i in 1..length {
            let magnitude = data[i].norm();
            assert!(magnitude < 1e-4,
                    "Bin {} magnitude {} should be near zero", i, magnitude);
        }
    }

    #[test]
    fn test_single_frequency() {
        // Test that a single frequency sine wave produces a peak at the correct bin
        let length = 128;
        let frequency_bin = 10; // We'll create a sine wave at bin 10

        let mut data: Vec<Complex32> = (0..length)
            .map(|i| {
                let t = i as f32 / length as f32;
                let phase = 2.0 * PI * frequency_bin as f32 * t;
                Complex::new(phase.cos(), phase.sin())
            })
            .collect();

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(length);
        fft.process(&mut data);

        // Find the bin with maximum magnitude
        let max_bin = data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(max_bin, frequency_bin,
                   "Peak should be at bin {}, found at bin {}", frequency_bin, max_bin);

        // The peak magnitude should be approximately equal to length
        let peak_magnitude = data[frequency_bin].norm();
        assert!((peak_magnitude - length as f32).abs() < 1.0,
                "Peak magnitude {} should be close to {}", peak_magnitude, length);
    }

    #[test]
    fn test_parsevals_theorem() {
        // Test Parseval's theorem: energy is conserved between time and frequency domains
        let length = 256;

        // Create a more complex signal (sum of multiple frequencies)
        let original: Vec<Complex32> = (0..length)
            .map(|i| {
                let t = i as f32 / length as f32;
                let sig = (2.0 * PI * 5.0 * t).cos() +
                         0.5 * (2.0 * PI * 13.0 * t).cos() +
                         0.3 * (2.0 * PI * 27.0 * t).sin();
                Complex::new(sig, 0.0)
            })
            .collect();

        // Calculate energy in time domain
        let time_energy: f32 = original.iter()
            .map(|x| x.norm_sqr())
            .sum();

        // Transform to frequency domain
        let mut freq_data = original.clone();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(length);
        fft.process(&mut freq_data);

        // Calculate energy in frequency domain
        let freq_energy: f32 = freq_data.iter()
            .map(|x| x.norm_sqr())
            .sum();

        // Parseval's theorem: sum(|x[n]|^2) = (1/N) * sum(|X[k]|^2)
        let normalized_freq_energy = freq_energy / length as f32;

        let relative_error = ((time_energy - normalized_freq_energy) / time_energy).abs();
        assert!(relative_error < 1e-5,
                "Energy conservation failed: time={}, freq/N={}, relative_error={}",
                time_energy, normalized_freq_energy, relative_error);
    }

    #[test]
    fn test_inverse_fft() {
        // Test that inverse FFT recovers the original signal
        let length = 128;

        // Create original signal
        let original: Vec<Complex32> = (0..length)
            .map(|i| {
                let t = i as f32 / length as f32;
                let sig = (2.0 * PI * 7.0 * t).cos() +
                         (2.0 * PI * 19.0 * t).sin();
                Complex::new(sig, 0.3 * (2.0 * PI * 3.0 * t).cos())
            })
            .collect();

        // Forward FFT
        let mut data = original.clone();
        let mut planner = FftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(length);
        fft_forward.process(&mut data);

        // Inverse FFT
        let fft_inverse = planner.plan_fft_inverse(length);
        fft_inverse.process(&mut data);

        // Normalize (FFT libraries typically don't normalize the inverse)
        for x in &mut data {
            *x /= length as f32;
        }

        // Compare with original
        for (i, (orig, recovered)) in original.iter().zip(data.iter()).enumerate() {
            let error = (orig - recovered).norm();
            assert!(error < 1e-4,
                    "Sample {} mismatch: original={:?}, recovered={:?}, error={}",
                    i, orig, recovered, error);
        }
    }

    #[test]
    fn test_batch_fft() {
        // Test that batch FFT processing works correctly
        let batch_size = 10;
        let length = 64;
        let total_size = batch_size * length;

        // Create batch data with different frequencies for each batch
        let mut data: Vec<Complex32> = (0..total_size)
            .map(|i| {
                let batch_idx = i / length;
                let sample_idx = i % length;
                let t = sample_idx as f32 / length as f32;
                let freq = (batch_idx + 1) as f32; // Each batch has a different frequency
                Complex::new((2.0 * PI * freq * t).cos(), (2.0 * PI * freq * t).sin())
            })
            .collect();

        // Perform batch FFT
        perform_batch_fft(&mut data, batch_size, length);

        // Verify each batch independently
        for batch_idx in 0..batch_size {
            let batch_start = batch_idx * length;
            let batch_data = &data[batch_start..batch_start + length];

            // Find peak bin
            let max_bin = batch_data.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            let expected_bin = batch_idx + 1;
            assert_eq!(max_bin, expected_bin,
                       "Batch {} peak should be at bin {}, found at bin {}",
                       batch_idx, expected_bin, max_bin);
        }
    }
}
