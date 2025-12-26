# Batch FFT - High-Performance 1D FFT Processor

A Rust application for performing batch 1D Complex-to-Complex (C2C) Fast Fourier Transforms with parallel processing and performance metrics.

## Features

- **Batch Processing**: Process multiple FFTs in parallel from a contiguous array
- **Multi-threaded**: Configurable thread count for optimal performance
- **Performance Metrics**: Real-time GFLOPS calculation and timing
- **Command-line Interface**: Easy-to-use CLI with argument parsing

## Requirements

- Rust 1.70 or later
- Cargo (Rust package manager)

## Installation

```bash
cargo build --release
```

## Usage

```bash
cargo run --release -- --batch <BATCH_SIZE> --length <FFT_LENGTH> --threads <NUM_THREADS>
```

### Arguments

- `--batch, -b`: Number of FFTs in the batch
- `--length, -l`: FFT transform length (number of samples per FFT)
- `--threads, -t`: Number of threads to use for parallel processing

### Example

Process 1000 FFTs of length 1024 using 8 threads:

```bash
cargo run --release -- --batch 1000 --length 1024 --threads 8
```

## Output

The application outputs results in CSV format for easy data processing:

Example output:
```
batch,fft_length,threads,time_ms,gflops
1000,1024,8,1.282,40
```

**CSV Fields:**
- `batch`: Number of FFTs in the batch
- `fft_length`: FFT transform length
- `threads`: Number of threads used
- `time_ms`: Execution time in milliseconds
- `gflops`: Performance in GFLOPS (billions of FLOPs per second)

## Performance Notes

- The FLOPS calculation uses the standard FFT complexity: `5*N*log2(N)` operations per FFT
- For batch processing: `Batch × 5 × N × log2(N)` total FLOPs
- Performance scales with thread count up to the number of physical cores
- Input data is stored in a contiguous array for optimal memory access

## Performance Comparison: Rust vs C++

A C++ implementation using FFTW3 with native threading is included in the `cpp-version/` directory for performance comparison. The C++ version uses `fftw_plan_many_dft()` for batch processing with `fftw_init_threads()` and `fftw_plan_with_nthreads()` for multi-threading. Plans are created with FFTW_MEASURE before timing measurements.

### Benchmark Results

All tests performed on the same hardware with identical workloads:

| Workload | Threads | Rust Time | Rust GFLOPS | C++ Time | C++ GFLOPS | Winner |
|----------|---------|-----------|-------------|----------|------------|--------|
| 100 × 256 | 4 | 0.104 ms | 10 | 0.062 ms | 17 | **C++ (1.7x)** |
| 1000 × 1024 | 8 | 1.501 ms | 34 | 1.743 ms | 29 | **Rust (1.2x)** |
| 2000 × 2048 | 8 | 4.270 ms | 53 | 5.307 ms | 42 | **Rust (1.2x)** |
| 2000 × 2048 | 1 | 13.335 ms | 17 | 15.253 ms | 15 | **Rust (1.1x)** |

### Key Findings

- **Performance is highly competitive** between Rust and C++/FFTW3
- **C++ wins on small workloads** (1.7x faster for 100×256), likely due to FFTW's optimized batch planning
- **Rust wins on medium-to-large workloads** (1.1-1.2x faster), demonstrating excellent scaling
- **Both implementations scale efficiently** with thread count
- **FFTW's native batch interface** (`fftw_plan_many_dft`) with threading is significantly more efficient than parallelizing individual FFTs with OpenMP (previous C++ implementation was 2-10x slower)
- **Both RustFFT and FFTW3 are world-class FFT libraries** with different performance characteristics

The results demonstrate that both Rust (with RustFFT and Rayon) and C++ (with FFTW3 native threading) can achieve excellent performance for batch FFT processing, with the optimal choice depending on workload size.

## Dependencies

- [rustfft](https://crates.io/crates/rustfft): High-performance FFT library
- [rayon](https://crates.io/crates/rayon): Data parallelism library
- [clap](https://crates.io/crates/clap): Command-line argument parser
- [num-complex](https://crates.io/crates/num-complex): Complex number support

## License

MIT
