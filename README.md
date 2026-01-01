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

## Performance Comparison: Rust vs C++ vs CUDA

Three implementations are included for comprehensive performance comparison:

1. **Rust** (this repository): Uses RustFFT with Rayon for CPU parallelism
2. **C++ FFTW** (`cpp-version/`): Uses FFTW3's `fftw_plan_many_dft()` with native threading
3. **CUDA** (`cuda/`): Uses NVIDIA cuFFT's `cufftPlanMany()` for GPU acceleration

### Benchmark Methodology

- **Thread Optimization**: Each test case uses the optimal thread count (1-8 threads) for best performance
- **Baseline**: C++ FFTW performance is the baseline (1.0x) for comparison
- **FFT Range**: 1K to 512K FFT sizes with batch sizes of 250-10,000
- **Timing**: Only FFT execution time is measured (excludes data generation and planning)

### Test System Specifications

- **CPU**: Intel Core i5-8400 @ 2.80GHz (6 cores, 2017)
- **GPU**: NVIDIA GeForce GTX 1080 (8GB VRAM, 2016)
- **OS**: Ubuntu 24.04.1 LTS
- **CUDA**: Version 12.9
- **FFTW**: Version 3.3.10

### Complete Benchmark Results (ALL with Fair Timing)

**Important**: All three implementations now use fair timing that excludes plan creation/destruction overhead. This ensures an apples-to-apples comparison.

| FFT Size | Batch | **Rust (CPU)** | **C++ FFTW (CPU)** | **CUDA (GPU)** | **Rust vs C++** | **CUDA vs C++** |
|----------|-------|----------------|-------------------|----------------|-----------------|-----------------|
| | | Threads / Time / GFLOPS | Threads / Time / GFLOPS | Time / GFLOPS | Speedup | Speedup |
| **1K** | 1000 | 4T / 0.92ms / 56 | 4T / 1.40ms / 37 | 0.20ms / 252 | **1.51x** | **6.81x** |
| **1K** | 10000 | 4T / 6.89ms / 74 | 8T / 12.09ms / 42 | 0.81ms / 628 | **1.76x** | **14.95x** |
| **2K** | 1000 | 4T / 1.59ms / 71 | 8T / 2.85ms / 39 | 0.28ms / 407 | **1.82x** | **10.44x** |
| **4K** | 1000 | 4T / 3.09ms / 80 | 8T / 5.74ms / 43 | 0.42ms / 588 | **1.86x** | **13.67x** |
| **8K** | 500 | 4T / 3.38ms / 79 | 4T / 5.91ms / 45 | 0.44ms / 607 | **1.76x** | **13.49x** |
| **16K** | 500 | 8T / 7.60ms / 75 | 8T / 15.68ms / 37 | 1.44ms / 399 | **2.03x** | **10.78x** |
| **32K** | 250 | 4T / 8.46ms / 73 | 8T / 16.77ms / 37 | 1.34ms / 460 | **1.97x** | **12.43x** |
| **64K** | 250 | 8T / 23.54ms / 56 | 4T / 39.84ms / 33 | 2.52ms / 521 | **1.70x** | **15.79x** |
| **128K** | 250 | 4T / 67.29ms / 41 | 4T / 117.08ms / 24 | 4.98ms / 559 | **1.71x** | **23.29x** |
| **256K** | 250 | 2T / 197.58ms / 30 | 4T / 295.73ms / 20 | 10.19ms / 579 | **1.50x** | **28.95x** |
| **512K** | 250 | 4T / 505.45ms / 25 | 8T / 747.96ms / 17 | 21.15ms / 589 | **1.47x** | **34.65x** |

### Key Findings

#### 1. **CPU Rust Dominates C++ FFTW** (Fair Timing)
- Rust is **1.74x faster on average** across all test cases
- **Wins all 11/11 test cases** against C++ FFTW
- Performance range: **1.47x - 2.03x faster**
- Peak Rust advantage: **2.03x at 16K FFT size**
- Rust achieves up to **80 GFLOPS** vs C++ max of 45 GFLOPS
- More efficient CPU parallelization across all workload sizes

#### 2. **CUDA Dominates with Massive GPU Speedup** ⚡
- **16.84x faster than C++ FFTW on average** (fair timing)
- **34.65x faster** than C++ at 512K FFT size (largest workload)
- **23-29x faster** for very large FFTs (128K-256K)
- **13-16x faster** for medium FFTs (8K-64K)
- **7-15x faster** even for small FFTs (1K-4K)
- Peak performance: **628 GFLOPS** vs C++ max of 45 GFLOPS

#### 3. **CUDA vs CPU Rust: GPU Wins All Cases**
- CUDA is **10.11x faster than Rust CPU on average** (fair timing)
- Speedup range: **4.50x - 23.56x** across all workloads
- **GPU wins all 11/11 test cases** against best CPU implementation
- For small FFTs (1K), CUDA achieves **252 GFLOPS** vs Rust's 56 GFLOPS
- For large FFTs (512K), CUDA achieves **589 GFLOPS** vs Rust's 25 GFLOPS

#### 4. **Optimal Thread Count Varies by Workload**
- Small FFTs (1K-8K): **4 threads** optimal for Rust
- Large FFTs (128K-512K): **2-4 threads** optimal (memory bandwidth limited)
- C++ FFTW: Mix of 4 and 8 threads depending on FFT size
- More threads ≠ better performance; sweet spot depends on workload

### Performance Summary

| Implementation | Peak GFLOPS | Average GFLOPS | Typical Range | Best Use Case |
|----------------|-------------|----------------|---------------|---------------|
| **CUDA (GPU)** | **628** | **508** | 250-628 | Any FFT workload - always fastest |
| **Rust (CPU)** | 80 | **60** | 25-80 | CPU-only scenarios, best CPU option |
| **C++ FFTW (CPU)** | 45 | **34** | 17-45 | Legacy compatibility only |

### Recommendations

**Choose CUDA when:**
- GPU is available (**7-35x faster** than C++, **4.5-24x faster** than Rust)
- Any FFT workload - CUDA dominates all 11 test cases
- Maximum performance is critical (up to **628 GFLOPS**)
- Processing any batch size or FFT size

**Choose Rust when:**
- No GPU available
- Need best CPU performance (**1.74x faster** than C++ on average)
- Consistent CPU performance across all workloads
- Memory safety and modern tooling are priorities

**Avoid C++ FFTW:**
- Rust CPU is faster in **all 11/11 test cases**
- CUDA is **7-35x faster** than C++ FFTW
- Use only for legacy code compatibility

### Conclusion

With **fair timing across all implementations** (all exclude plan creation), the benchmarks demonstrate three distinct performance tiers:

1. **CUDA (GPU)**: Dominant across all workloads - **7-35x faster** than C++, **4.5-24x faster** than Rust CPU
   - Average: **508 GFLOPS**, Peak: 628 GFLOPS
2. **Rust (CPU)**: Best CPU implementation - **1.74x faster** than C++ on average, **wins all 11/11 cases**
   - Average: **60 GFLOPS**, Peak: 80 GFLOPS
3. **C++ FFTW (CPU)**: Baseline reference - slowest of all three
   - Average: **34 GFLOPS**, Peak: 45 GFLOPS

**Critical Timing Methodology Fix**: All three implementations now use fair timing that excludes plan creation/destruction:
- **CUDA C++**: Plan creation moved outside timing (originally included)
- **Rust CPU**: Plan creation moved outside timing (originally included)
- **C++ FFTW**: Already correct (plan created before timing)

This fair comparison reveals CUDA's true 10-17x performance advantage over CPU implementations.

**For new projects**: Use **CUDA** whenever a GPU is available, **Rust** for CPU-only deployments.

## Dependencies

- [rustfft](https://crates.io/crates/rustfft): High-performance FFT library
- [rayon](https://crates.io/crates/rayon): Data parallelism library
- [clap](https://crates.io/crates/clap): Command-line argument parser
- [num-complex](https://crates.io/crates/num-complex): Complex number support

## License

MIT
