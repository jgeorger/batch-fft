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

A C++ implementation using FFTW3 and OpenMP is included in the `cpp-version/` directory for performance comparison. The C++ version uses FFTW_PATIENT planning and creates FFT plans before timing measurements.

### Benchmark Results

All tests performed on the same hardware with identical workloads:

| Workload | Threads | Rust Time | Rust GFLOPS | C++ Time | C++ GFLOPS | Rust Speedup |
|----------|---------|-----------|-------------|----------|------------|--------------|
| 100 × 256 | 4 | 0.104 ms | 10 | 0.665 ms | 2 | **6.4x** |
| 1000 × 1024 | 8 | 1.282 ms | 40 | 3.387 ms | 15 | **2.6x** |
| 2000 × 2048 | 8 | 3.208 ms | 70 | 5.829 ms | 39 | **1.8x** |
| 2000 × 2048 | 1 | 13.515 ms | 17 | 18.134 ms | 12 | **1.3x** |

### Key Findings

- **Rust outperforms C++/FFTW3 by 1.3x to 6.4x** across all workloads
- **Largest advantage on small workloads** (6.4x faster for 100×256), suggesting lower overhead
- **Both scale well with threading**, but Rust maintains its performance advantage
- **Even single-threaded, Rust is 1.3x faster**, demonstrating superior core FFT implementation
- **RustFFT library is highly competitive** with the industry-standard FFTW3, and in these tests, significantly faster

The performance advantage demonstrates the quality of the RustFFT library and the efficiency of Rayon's parallel execution model.

## Dependencies

- [rustfft](https://crates.io/crates/rustfft): High-performance FFT library
- [rayon](https://crates.io/crates/rayon): Data parallelism library
- [clap](https://crates.io/crates/clap): Command-line argument parser
- [num-complex](https://crates.io/crates/num-complex): Complex number support

## License

MIT
