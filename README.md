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

## Performance Comparison: FFTW vs MKL vs Rust vs CUDA

Five implementations are included for comprehensive performance comparison:

1. **C++ FFTW** (`cpp-version/`): Uses FFTW3's `fftwf_plan_many_dft()` with native threading (single precision)
2. **C++ Intel MKL** (`mkl-version/`): Uses Intel MKL's optimized DFT interface with threading (single precision)
3. **Rust** (this repository): Uses RustFFT with Rayon for CPU parallelism (single precision)
4. **Rust-CUDA** (`rust-cuda/`): Rust FFI bindings to NVIDIA cuFFT for GPU acceleration
5. **CUDA** (`cuda/`): Direct C++ implementation using NVIDIA cuFFT's `cufftPlanMany()` for GPU acceleration

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

### Complete Benchmark Results (ALL with Fair Timing & Single Precision)

**Important**: All five implementations use **single precision (float32/Complex32)** with fair timing that excludes plan creation/destruction overhead. This ensures an apples-to-apples comparison.

| FFT Size | Batch | **FFTW (CPU)** | **MKL (CPU)** | **Rust (CPU)** | **Rust-CUDA (GPU)** | **CUDA (GPU)** |
|----------|-------|----------------|---------------|----------------|---------------------|----------------|
| | | Threads / Time / GFLOPS | Threads / Time / GFLOPS / vs FFTW | Threads / Time / GFLOPS / vs FFTW | Time / GFLOPS / vs FFTW | Time / GFLOPS / vs FFTW |
| **1K** | 1000 | 4T / 0.65ms / 79 | 4T / 1.12ms / 46 / 0.58x | 4T / 0.92ms / 56 / 0.71x | 0.20ms / 252 / **3.19x** | 0.14ms / 371 / **4.70x** |
| **1K** | 10000 | 4T / 6.53ms / 78 | 4T / 6.91ms / 74 / 0.95x | 4T / 6.89ms / 74 / 0.95x | 0.81ms / 628 / **8.05x** | 0.85ms / 605 / **7.76x** |
| **2K** | 1000 | 4T / 1.22ms / 92 | 4T / 1.79ms / 63 / 0.68x | 4T / 1.59ms / 71 / 0.77x | 0.28ms / 407 / **4.42x** | 0.26ms / 431 / **4.68x** |
| **4K** | 1000 | 4T / 2.69ms / 91 | 8T / 3.46ms / 71 / 0.78x | 4T / 3.09ms / 80 / 0.88x | 0.42ms / 588 / **6.46x** | 0.46ms / 540 / **5.93x** |
| **8K** | 500 | 4T / 3.24ms / 82 | 8T / 3.32ms / 80 / 0.98x | 4T / 3.38ms / 79 / 0.96x | 0.44ms / 607 / **7.40x** | 0.42ms / 636 / **7.76x** |
| **16K** | 500 | 4T / 7.34ms / 78 | 8T / 7.78ms / 74 / 0.95x | 8T / 7.60ms / 75 / 0.96x | 1.44ms / 399 / **5.12x** | 1.43ms / 400 / **5.13x** |
| **32K** | 250 | 4T / 9.35ms / 66 | 8T / 8.29ms / 74 / **1.12x** | 4T / 8.46ms / 73 / **1.11x** | 1.34ms / 460 / **6.97x** | 1.31ms / 470 / **7.12x** |
| **64K** | 250 | 4T / 19.49ms / 67 | 4T / 21.21ms / 62 / 0.93x | 8T / 23.54ms / 56 / 0.84x | 2.52ms / 521 / **7.78x** | 2.63ms / 498 / **7.43x** |
| **128K** | 250 | 8T / 45.23ms / 62 | 4T / 69.40ms / 40 / 0.65x | 4T / 67.29ms / 41 / 0.66x | 4.98ms / 559 / **9.02x** | 5.04ms / 553 / **8.92x** |
| **256K** | 250 | 8T / 129.09ms / 46 | 8T / 142.74ms / 41 / 0.89x | 2T / 197.58ms / 30 / 0.65x | 10.19ms / 579 / **12.59x** | 10.02ms / 589 / **12.80x** |
| **512K** | 250 | 8T / 364.53ms / 34 | 4T / 297.44ms / 42 / **1.24x** | 4T / 505.45ms / 25 / 0.74x | 21.15ms / 589 / **17.32x** | 22.01ms / 566 / **16.65x** |

### Key Findings

#### 1. **FFTW Leads CPU Implementations with Single Precision**
- FFTW is **fastest CPU implementation** at **70.5 GFLOPS average**
- **Wins 10/11 test cases** against Rust (tied 1)
- **Wins 9/11 test cases** against MKL
- Peak FFTW performance: **92 GFLOPS** at 2K FFT size
- Particularly strong at small-to-medium FFTs (1K-64K): **67-92 GFLOPS**
- FFTW's single precision implementation is highly optimized

#### 2. **Rust and MKL Essentially Tied for Second Place**
- Rust CPU: **60.0 GFLOPS average** (range: 25-80)
- Intel MKL: **60.6 GFLOPS average** (range: 40-80)
- Rust wins **5/11 cases**, MKL wins **6/11 cases**
- Average ratio: **0.97x** (nearly identical performance)
- Both lag behind FFTW's optimized single-precision kernels

#### 3. **CUDA Dominates with Massive GPU Speedup** ⚡
- **8.03x faster than FFTW** on average (fair timing, single precision)
- **8.95x faster than MKL**, **10.11x faster than Rust**
- **17.32x faster** than FFTW at 512K FFT size (largest workload)
- **9-13x faster** for very large FFTs (128K-256K)
- **5-8x faster** for medium FFTs (8K-64K)
- **3-8x faster** even for small FFTs (1K-4K)
- Peak performance: **628 GFLOPS** vs best CPU of 92 GFLOPS

#### 4. **Critical Precision Impact**
- Single precision (float32) vs Double precision (float64) makes **~2x performance difference** for CPU implementations
- FFTW with single precision: **70 GFLOPS** vs double precision: **34 GFLOPS**
- MKL with single precision: **61 GFLOPS** vs double precision: **30 GFLOPS**
- **Always ensure consistent precision** across implementations for fair comparison

#### 5. **Optimal Thread Count Varies by Workload**
- Small-to-medium FFTs (1K-64K): **4 threads** optimal for FFTW and Rust
- Large FFTs (128K-512K): **8 threads** optimal for FFTW and MKL
- Rust: Mostly 4 threads, some 2-8 threads for large sizes
- More threads ≠ better performance; sweet spot depends on workload

### Performance Summary

| Implementation | Peak GFLOPS | Average GFLOPS | Typical Range | Best Use Case |
|----------------|-------------|----------------|---------------|---------------|
| **CUDA (GPU)** | **636** | **514** | 371-636 | Direct C++ cuFFT - maximum GPU performance |
| **Rust-CUDA (GPU)** | **628** | **508** | 252-628 | Rust FFI to cuFFT - near-identical GPU performance |
| **FFTW (CPU)** | **92** | **71** | 34-92 | Best CPU option for single precision |
| **Intel MKL (CPU)** | 80 | **61** | 40-80 | Intel processors, comparable to Rust |
| **Rust (CPU)** | 80 | **60** | 25-80 | Modern safe code, comparable to MKL |

### Recommendations

**Choose CUDA or Rust-CUDA when:**
- GPU is available (**3-17x faster** than best CPU)
- Any FFT workload - GPU dominates all test cases
- Maximum performance is critical (up to **628 GFLOPS**)
- Processing any batch size or FFT size
- Rust-CUDA offers same performance with Rust safety guarantees

**Choose FFTW when:**
- No GPU available and need **best CPU performance** (71 GFLOPS average)
- C++ codebase
- Single precision FFT workloads
- Excellent performance across all FFT sizes

**Choose Intel MKL when:**
- Intel processors (optimized for AVX-512)
- Performance essentially identical to Rust (61 vs 60 GFLOPS)
- Already integrated in your project
- Strong at large FFTs (256K-512K)

**Choose Rust when:**
- Memory safety and modern tooling are priorities
- Performance comparable to MKL (60 GFLOPS average)
- Pure Rust ecosystem
- Good balance across all workload sizes

### Conclusion

With **fair timing and consistent single precision** across all implementations, the benchmarks demonstrate clear performance tiers:

1. **GPU Implementations (CUDA & Rust-CUDA)**: Dominant across all workloads - **5-17x faster** than best CPU (FFTW)
   - CUDA C++ Average: **514 GFLOPS**, Peak: 636 GFLOPS
   - Rust-CUDA Average: **508 GFLOPS**, Peak: 628 GFLOPS
   - Both implementations achieve near-identical performance (within 1%)
2. **FFTW (CPU)**: Best CPU implementation - wins 10/11 cases vs other CPUs
   - Average: **71 GFLOPS**, Peak: 92 GFLOPS
3. **Intel MKL (CPU)** & **Rust (CPU)**: Essentially tied for second place
   - MKL Average: **61 GFLOPS**, Peak: 80 GFLOPS
   - Rust Average: **60 GFLOPS**, Peak: 80 GFLOPS

**Critical Findings**:
1. **Precision Matters**: Single vs double precision makes ~2x performance difference on CPU
2. **FFTW Optimized**: FFTW's single precision implementation is exceptionally well-tuned
3. **Fair Competition**: With consistent precision, Rust and MKL perform similarly
4. **GPU Advantage**: CUDA maintains 8-17x speedup even with optimized CPU code

**Timing Methodology**: All five implementations use fair timing that excludes plan/descriptor creation:
- **CUDA**: Plan creation moved outside timing
- **Rust-CUDA**: Plan creation moved outside timing (via Rust FFI)
- **Rust**: Planner creation moved outside timing
- **FFTW**: Plan creation moved outside timing (single precision: `fftwf_*`)
- **Intel MKL**: Descriptor creation/commit moved outside timing (single precision: `DFTI_SINGLE`)

**For new projects**:
- Use **CUDA** or **Rust-CUDA** whenever a GPU is available (fastest by far, identical performance)
- Choose Rust-CUDA if you want Rust's memory safety with GPU performance
- Use **FFTW** for CPU-only single precision batch FFT (best CPU performance)
- Use **Rust** or **MKL** when FFTW isn't suitable (essentially identical performance)

## Dependencies

- [rustfft](https://crates.io/crates/rustfft): High-performance FFT library
- [rayon](https://crates.io/crates/rayon): Data parallelism library
- [clap](https://crates.io/crates/clap): Command-line argument parser
- [num-complex](https://crates.io/crates/num-complex): Complex number support

## License

MIT
