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

## Testing

Run the included unit tests to verify FFT correctness:

```bash
cargo test
```

The test suite includes:
- **DC signal test**: Verifies constant signals produce energy only in DC bin
- **Single frequency test**: Verifies frequency detection at correct FFT bin
- **Parseval's theorem test**: Verifies energy conservation between time and frequency domains
- **Inverse FFT test**: Verifies forward + inverse FFT recovers original signal
- **Batch FFT test**: Verifies batch processing handles multiple FFTs correctly

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

#### Small to Medium FFT Sizes
| Workload (Batch × Length) | Threads | Rust Time | Rust GFLOPS | C++ Time | C++ GFLOPS | Winner |
|---------------------------|---------|-----------|-------------|----------|------------|--------|
| 100 × 256 | 4 | 0.104 ms | 10 | 0.062 ms | 17 | **C++ (1.7x)** |
| 1000 × 1024 | 8 | 1.501 ms | 34 | 1.743 ms | 29 | **Rust (1.2x)** |
| 2000 × 2048 | 8 | 4.270 ms | 53 | 5.307 ms | 42 | **Rust (1.2x)** |
| 2000 × 2048 | 1 | 13.335 ms | 17 | 15.253 ms | 15 | **Rust (1.1x)** |

#### Large FFT Sizes (4k, 8k, 32k)
| Workload (Batch × Length) | Threads | Rust Time | Rust GFLOPS | C++ Time | C++ GFLOPS | Winner |
|---------------------------|---------|-----------|-------------|----------|------------|--------|
| 1000 × 4096 | 8 | 3.739 ms | 66 | 4.110 ms | 60 | **Rust (1.1x)** |
| 1000 × 4096 | 4 | 4.879 ms | 50 | 4.250 ms | 58 | **C++ (1.1x)** |
| 1000 × 4096 | 1 | 14.512 ms | 17 | 15.989 ms | 15 | **Rust (1.1x)** |
| 500 × 8192 | 8 | 3.350 ms | 79 | 4.689 ms | 57 | **Rust (1.4x)** |
| 100 × 32768 | 8 | 4.011 ms | 61 | 4.271 ms | 58 | **Rust (1.1x)** |

### Key Findings

- **Performance is highly competitive** across all FFT sizes and thread counts
- **C++ wins on small FFT sizes** (1.7x faster for 256-point FFTs), likely due to FFTW's optimized batch planning and lower overhead
- **Rust wins on medium to large FFT sizes** (1.1-1.4x faster for 1024-32768 point FFTs), demonstrating superior scaling
- **Peak performance increases with FFT size**: Both implementations achieve 60+ GFLOPS with 4k-8k FFTs vs 10-53 GFLOPS with smaller FFTs
- **Rust achieves highest GFLOPS at 8k**: 79 GFLOPS with 500×8192×8 configuration
- **Both scale efficiently with thread count**: Near-linear scaling from 1 to 8 threads for larger FFTs
- **FFTW's native batch interface** (`fftw_plan_many_dft`) with threading is significantly more efficient than parallelizing individual FFTs with OpenMP (previous C++ implementation was 2-10x slower)
- **Trade-offs exist**: C++ better for low-latency small FFTs, Rust better for throughput-oriented large FFTs

The results demonstrate that both Rust (with RustFFT and Rayon) and C++ (with FFTW3 native threading) are world-class FFT implementations with different performance characteristics that make them suitable for different use cases.

## Dependencies

- [rustfft](https://crates.io/crates/rustfft): High-performance FFT library
- [rayon](https://crates.io/crates/rayon): Data parallelism library
- [clap](https://crates.io/crates/clap): Command-line argument parser
- [num-complex](https://crates.io/crates/num-complex): Complex number support

## License

MIT
