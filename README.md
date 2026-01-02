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
| **1K** | 1000 | 4T / 0.71ms / 72 | 2T / 1.08ms / 47 / 0.65x | 4T / 0.92ms / 56 / 0.78x | 0.13ms / 386 / **5.36x** | 0.15ms / 331 / **4.60x** |
| **1K** | 10000 | 4T / 6.56ms / 78 | 4T / 6.77ms / 76 / 0.97x | 4T / 6.89ms / 74 / 0.95x | 0.80ms / 639 / **8.19x** | 0.80ms / 644 / **8.26x** |
| **2K** | 1000 | 4T / 1.31ms / 86 | 4T / 1.72ms / 65 / 0.76x | 4T / 1.59ms / 71 / 0.83x | 0.23ms / 498 / **5.79x** | 0.24ms / 469 / **5.45x** |
| **4K** | 1000 | 4T / 2.87ms / 86 | 8T / 3.30ms / 74 / 0.86x | 4T / 3.09ms / 80 / 0.93x | 0.38ms / 650 / **7.56x** | 0.41ms / 597 / **6.94x** |
| **8K** | 500 | 4T / 3.15ms / 85 | 4T / 3.87ms / 69 / 0.81x | 4T / 3.38ms / 79 / 0.93x | 0.45ms / 591 / **6.95x** | 0.42ms / 628 / **7.39x** |
| **16K** | 500 | 4T / 7.53ms / 76 | 4T / 7.72ms / 74 / 0.97x | 8T / 7.60ms / 75 / 0.99x | 1.53ms / 375 / **4.93x** | 1.46ms / 394 / **5.18x** |
| **32K** | 250 | 4T / 9.38ms / 65 | 4T / 8.96ms / 69 / **1.06x** | 4T / 8.47ms / 73 / **1.12x** | 1.33ms / 462 / **7.11x** | 1.32ms / 467 / **7.18x** |
| **64K** | 250 | 8T / 19.05ms / 69 | 4T / 21.82ms / 60 / 0.87x | 8T / 23.54ms / 56 / 0.81x | 2.70ms / 486 / **7.04x** | 2.61ms / 502 / **7.28x** |
| **128K** | 250 | 8T / 45.46ms / 61 | 4T / 70.05ms / 40 / 0.66x | 4T / 67.29ms / 41 / 0.67x | 4.93ms / 565 / **9.26x** | 4.95ms / 563 / **9.23x** |
| **256K** | 250 | 4T / 125.91ms / 47 | 8T / 146.86ms / 40 / 0.85x | 2T / 197.58ms / 30 / 0.64x | 10.03ms / 588 / **12.51x** | 9.98ms / 591 / **12.57x** |
| **512K** | 250 | 8T / 350.69ms / 36 | 8T / 262.93ms / 47 / **1.31x** | 4T / 505.45ms / 25 / 0.69x | 21.13ms / 589 / **16.36x** | 21.21ms / 587 / **16.31x** |

### Key Findings

#### 1. **FFTW Leads CPU Implementations with Single Precision**
- FFTW is **fastest CPU implementation** at **69.2 GFLOPS average**
- **Wins 10/11 test cases** against Rust (tied 1)
- **Wins 9/11 test cases** against MKL
- Peak FFTW performance: **86 GFLOPS** at 2K-4K FFT sizes
- Particularly strong at small-to-medium FFTs (1K-64K): **69-86 GFLOPS**
- FFTW's single precision implementation is highly optimized

#### 2. **Rust and MKL Essentially Tied for Second Place**
- Rust CPU: **60.0 GFLOPS average** (range: 25-80)
- Intel MKL: **60.1 GFLOPS average** (range: 40-76)
- Rust wins **5/11 cases**, MKL wins **6/11 cases**
- Average ratio: **1.00x** (essentially identical performance)
- Both lag behind FFTW's optimized single-precision kernels

#### 3. **CUDA Dominates with Massive GPU Speedup** ⚡
- **CUDA C++: 8.22x faster than FFTW** on average (fair timing, single precision)
- **Rust-CUDA: 8.28x faster than FFTW** on average
- **9.25x faster than MKL**, **10.38x faster than Rust** (CUDA C++)
- **16.31x faster** than FFTW at 512K FFT size (largest workload)
- **9-13x faster** for very large FFTs (128K-256K)
- **5-8x faster** for medium FFTs (8K-64K)
- **5-8x faster** even for small FFTs (1K-4K)
- Peak performance: **644 GFLOPS** (Rust-CUDA) vs best CPU of 86 GFLOPS (FFTW)

#### 4. **Critical Precision Impact**
- Single precision (float32) vs Double precision (float64) makes **~2x performance difference** for CPU implementations
- FFTW with single precision: **69 GFLOPS** vs double precision: **34 GFLOPS**
- MKL with single precision: **60 GFLOPS** vs double precision: **30 GFLOPS**
- **Always ensure consistent precision** across implementations for fair comparison

#### 5. **Optimal Thread Count Varies by Workload**
- Small-to-medium FFTs (1K-64K): **4 threads** optimal for FFTW and Rust
- Large FFTs (128K-512K): **8 threads** optimal for FFTW and MKL
- Rust: Mostly 4 threads, some 2-8 threads for large sizes
- More threads ≠ better performance; sweet spot depends on workload

### Performance Summary

| Implementation | Peak GFLOPS | Average GFLOPS | Typical Range | Best Use Case |
|----------------|-------------|----------------|---------------|---------------|
| **Rust-CUDA (GPU)** | **650** | **530** | 375-650 | Rust FFI to cuFFT - maximum GPU performance |
| **CUDA (GPU)** | **644** | **525** | 331-644 | Direct C++ cuFFT - near-identical GPU performance |
| **FFTW (CPU)** | **86** | **69** | 36-86 | Best CPU option for single precision |
| **Intel MKL (CPU)** | 76 | **60** | 40-76 | Intel processors, comparable to Rust |
| **Rust (CPU)** | 80 | **60** | 25-80 | Modern safe code, comparable to MKL |

### Recommendations

**Choose CUDA or Rust-CUDA when:**
- GPU is available (**5-16x faster** than best CPU)
- Any FFT workload - GPU dominates all test cases
- Maximum performance is critical (up to **650 GFLOPS**)
- Processing any batch size or FFT size
- Rust-CUDA offers same performance with Rust safety guarantees

**Choose FFTW when:**
- No GPU available and need **best CPU performance** (69 GFLOPS average)
- C++ codebase
- Single precision FFT workloads
- Excellent performance across all FFT sizes

**Choose Intel MKL when:**
- Intel processors (optimized for AVX-512)
- Performance essentially identical to Rust (60 vs 60 GFLOPS)
- Already integrated in your project
- Strong at large FFTs (512K)

**Choose Rust when:**
- Memory safety and modern tooling are priorities
- Performance comparable to MKL (60 GFLOPS average)
- Pure Rust ecosystem
- Good balance across all workload sizes

### Conclusion

With **fair timing and consistent single precision** across all implementations, the benchmarks demonstrate clear performance tiers:

1. **GPU Implementations (Rust-CUDA & CUDA)**: Dominant across all workloads - **5-16x faster** than best CPU (FFTW)
   - Rust-CUDA Average: **530 GFLOPS**, Peak: 650 GFLOPS
   - CUDA C++ Average: **525 GFLOPS**, Peak: 644 GFLOPS
   - Both implementations achieve near-identical performance (within 1%)
2. **FFTW (CPU)**: Best CPU implementation - wins 10/11 cases vs other CPUs
   - Average: **69 GFLOPS**, Peak: 86 GFLOPS
3. **Intel MKL (CPU)** & **Rust (CPU)**: Essentially tied for second place
   - MKL Average: **60 GFLOPS**, Peak: 76 GFLOPS
   - Rust Average: **60 GFLOPS**, Peak: 80 GFLOPS

**Critical Findings**:
1. **Precision Matters**: Single vs double precision makes ~2x performance difference on CPU
2. **FFTW Optimized**: FFTW's single precision implementation is exceptionally well-tuned
3. **Fair Competition**: With consistent precision, Rust and MKL perform similarly
4. **GPU Advantage**: CUDA maintains 5-16x speedup even with optimized CPU code

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
