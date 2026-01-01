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

### Complete Benchmark Results

| FFT Size | Batch | **Rust (Optimized)** | **C++ FFTW (Baseline)** | **CUDA** | **Rust vs C++** | **CUDA vs C++** |
|----------|-------|---------------------|------------------------|----------|-----------------|-----------------|
| | | Threads / Time / GFLOPS | Threads / Time / GFLOPS | Time / GFLOPS | Speedup | Speedup |
| **1K** | 1000 | 4T / 0.82ms / 62 | 4T / 1.22ms / 42 | 1.43ms / 36 | **1.47x** | 0.84x |
| **1K** | 10000 | 4T / 6.74ms / 76 | 8T / 11.51ms / 44 | 2.50ms / 205 | **1.70x** | **4.61x** |
| **2K** | 1000 | 4T / 1.53ms / 73 | 8T / 2.65ms / 43 | 1.61ms / 70 | **1.72x** | **1.64x** |
| **4K** | 1000 | 4T / 3.47ms / 71 | 8T / 5.50ms / 45 | 1.85ms / 133 | **1.58x** | **2.97x** |
| **8K** | 500 | 4T / 3.21ms / 83 | 4T / 5.59ms / 48 | 2.03ms / 131 | **1.74x** | **2.75x** |
| **16K** | 500 | 8T / 7.63ms / 75 | 8T / 14.45ms / 40 | 6.62ms / 87 | **1.89x** | **2.18x** |
| **32K** | 250 | 4T / 9.12ms / 67 | 8T / 16.50ms / 37 | 6.17ms / 100 | **1.80x** | **2.67x** |
| **64K** | 250 | 8T / 23.13ms / 57 | 4T / 39.50ms / 33 | 10.21ms / 128 | **1.70x** | **3.86x** |
| **128K** | 250 | 4T / 65.69ms / 42 | 4T / 118.21ms / 24 | 17.70ms / 157 | **1.79x** | **6.67x** |
| **256K** | 250 | 2T / 203.99ms / 29 | 4T / 303.71ms / 19 | 31.04ms / 190 | **1.48x** | **9.78x** |
| **512K** | 250 | 4T / 503.03ms / 25 | 8T / 733.58ms / 17 | 62.10ms / 201 | **1.45x** | **11.81x** |

### Key Findings

#### 1. **Rust Dominates CPU Performance**
- **Rust outperforms C++ FFTW in every single test case** (1.45x - 1.89x faster)
- Peak Rust advantage: **1.89x at 16K FFT size**
- Rust achieves up to **83 GFLOPS** vs C++ max of 48 GFLOPS
- More efficient CPU parallelization across all workload sizes

#### 2. **CUDA Provides Massive Speedup for Large FFTs**
- **11.81x faster** than C++ at 512K FFT size
- **6.67x - 9.78x faster** for 128K-256K FFTs
- **2.18x - 3.86x faster** for medium FFTs (16K-64K)
- Only slower (0.84x) for small batches of small FFTs (overhead-dominated)

#### 3. **Optimal Thread Count Varies by Workload**
- Small FFTs (1K-8K): **4 threads** optimal for Rust
- Large FFTs (128K-512K): **2-4 threads** optimal (memory bandwidth limited)
- C++ FFTW: Mix of 4 and 8 threads depending on FFT size
- More threads ≠ better performance; sweet spot depends on workload

#### 4. **GPU Batch Processing Scales Exceptionally**
- Small batch (1K × 1000): 36 GFLOPS
- Large batch (1K × 10000): **205 GFLOPS** (5.7x improvement)
- Large FFTs maintain 190-201 GFLOPS sustained performance
- GPU needs sufficient parallelism to reach peak efficiency

### Performance Summary

| Implementation | Peak GFLOPS | Typical Range | Best Use Case |
|----------------|-------------|---------------|---------------|
| **CUDA** | **205** | 100-200 | Large FFTs (≥4K) or large batches |
| **Rust** | 83 | 60-75 | CPU-only scenarios, all workloads |
| **C++ FFTW** | 48 | 20-45 | Legacy compatibility only |

### Recommendations

**Choose CUDA when:**
- FFT size ≥ 2K (**1.6x - 11.8x faster** than C++)
- Large batches with small FFTs (**4.6x faster** for 10K×1K)
- Maximum performance is critical (up to 205 GFLOPS)

**Choose Rust when:**
- No GPU available
- Need best CPU performance (**1.5x - 1.9x faster** than C++)
- Consistent performance across all workloads

**Avoid C++ FFTW:**
- Rust is faster in every test case
- Use only for legacy code compatibility

### Conclusion

The benchmark results demonstrate three distinct performance tiers:

1. **CUDA (GPU)**: Dominant for production workloads - up to **11.8x faster** than C++
2. **Rust (CPU)**: Best CPU implementation - consistently **1.5-1.9x faster** than C++
3. **C++ FFTW (CPU)**: Baseline reference - slower than both alternatives

For new projects: use **CUDA** for maximum performance, **Rust** for CPU-only deployments.

## Dependencies

- [rustfft](https://crates.io/crates/rustfft): High-performance FFT library
- [rayon](https://crates.io/crates/rayon): Data parallelism library
- [clap](https://crates.io/crates/clap): Command-line argument parser
- [num-complex](https://crates.io/crates/num-complex): Complex number support

## License

MIT
