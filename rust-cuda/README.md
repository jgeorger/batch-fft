# Rust FFI Batch FFT with cuFFT

A Rust implementation of batch 1D Complex-to-Complex FFT using direct FFI bindings to NVIDIA cuFFT.

## Overview

This implementation demonstrates:
- **Pure FFI approach** - Direct bindings to CUDA runtime and cuFFT (no wrapper libraries)
- **Safe Rust abstractions** - RAII wrappers around unsafe FFI calls
- **Performance parity** - Matches CUDA C++ performance when measuring comparable operations
- **Clean separation** - Setup costs (plan creation) separated from execution timing

## Features

- Direct FFI bindings to cuFFT library
- Safe RAII wrappers (`CudaBuffer`, `CudaEvent`, `CufftPlan`)
- Batch FFT processing on GPU
- CUDA event-based timing
- CSV output format for benchmarking

## Requirements

- Rust 1.70 or later
- CUDA Toolkit 11.0 or later
- NVIDIA GPU with compute capability 6.0+

## Building

```bash
cargo build --release
```

The build system will automatically link to:
- `libcufft` (cuFFT library)
- `libcudart` (CUDA runtime)

Library search paths:
- `/usr/local/cuda/lib64`
- `/usr/lib/x86_64-linux-gnu`
- `$CUDA_PATH/lib64` (if CUDA_PATH environment variable is set)

## Usage

```bash
./target/release/batch_fft -b <BATCH_SIZE> -l <FFT_LENGTH>
```

### Arguments

- `-b, --batch <SIZE>` - Number of FFTs in the batch
- `-l, --length <SIZE>` - FFT transform length (samples per FFT)

### Example

```bash
./target/release/batch_fft -b 1000 -l 1024
```

## Output

CSV format:
```
batch,fft_length,time_ms,gflops
1000,1024,0.133,386
```

**Fields:**
- `batch` - Number of FFTs processed
- `fft_length` - Transform length
- `time_ms` - Execution time in milliseconds (FFT only, excludes plan creation)
- `gflops` - Performance in GFLOPS

## Benchmarking

Run comprehensive benchmarks:

```bash
cd examples
./benchmark.sh
```

This will test 11 configurations from 1K to 512K FFT sizes and generate `rust_cuda_results.csv`.

## Performance Comparison

See [FAIR_COMPARISON.md](./FAIR_COMPARISON.md) for the complete performance analysis.

**Result**: ✅ **Performance parity achieved!** Both implementations perform within 4.6% of each other.

- **CUDA C++ average**: 508 GFLOPS
- **Rust FFI average**: 530 GFLOPS
- **Difference**: 4.6% (Rust slightly faster, within measurement variance)

Both implementations use the same timing methodology (excluding plan creation/destruction), demonstrating that Rust FFI has minimal overhead when calling CUDA libraries.

## Project Structure

```
rust-cuda/
├── Cargo.toml              # Rust package configuration
├── build.rs                # Build script (links CUDA libraries)
├── src/
│   ├── main.rs             # Main entry point
│   ├── cuda_ffi.rs         # CUDA runtime FFI bindings
│   └── cufft_ffi.rs        # cuFFT FFI bindings
├── examples/
│   ├── benchmark.sh        # Benchmark automation script
│   └── compare_results.py  # Results comparison tool
├── README.md               # This file
└── ANALYSIS.md             # Performance analysis
```

## Implementation Details

### CUDA Runtime FFI (`cuda_ffi.rs`)

Safe wrappers around CUDA runtime:
- **`CudaBuffer<T>`** - Device memory allocation with automatic deallocation
- **`CudaEvent`** - CUDA event for GPU timing
- **FFI functions**: cudaMalloc, cudaFree, cudaMemcpy, cudaEventCreate, etc.

### cuFFT FFI (`cufft_ffi.rs`)

Safe wrappers around cuFFT:
- **`CufftPlan`** - FFT plan with automatic cleanup on drop
- **FFI functions**: cufftPlanMany, cufftExecC2C, cufftDestroy
- **Error handling**: Convert cufftResult to Rust Result types

### Main Implementation (`main.rs`)

1. Parse command-line arguments
2. Generate input data (sine waves with varying frequencies)
3. Allocate device memory
4. Create cuFFT plan (BEFORE timing)
5. Warm-up run (excluded from timing)
6. Timed run with CUDA events (FFT execution only)
7. Calculate performance metrics
8. Output CSV results

## Timing Methodology

**What is timed:**
- ✅ FFT execution (cufftExecC2C)
- ✅ Device synchronization
- ❌ Plan creation (cufftPlanMany) - excluded
- ❌ Plan destruction (cufftDestroy) - excluded
- ❌ Memory allocation/copying - excluded

This matches best practices for benchmarking FFT libraries where setup costs are amortized across many operations.

## Comparison with CUDA C++

| Aspect | CUDA C++ | Rust FFI |
|--------|----------|----------|
| **Language** | C++ | Rust |
| **cuFFT Binding** | Direct C API | FFI bindings |
| **Memory Safety** | Manual | RAII wrappers (automatic cleanup) |
| **Plan Timing** | Included in benchmark | Excluded from benchmark |
| **FFT Performance** | ~200-650 GFLOPS | ~200-650 GFLOPS (same) |

## Performance Results Summary

Average across 11 test configurations:
- **Rust FFI**: 200-650 GFLOPS (FFT execution only)
- **CUDA C++**: 30-207 GFLOPS (includes plan creation overhead)

When normalized for plan creation overhead:
- **Both achieve ~200-650 GFLOPS** for pure FFT execution
- Performance varies with FFT size and batch count
- GPU memory bandwidth is the limiting factor

## Known Limitations

- Single precision only (Complex32 / float)
- Forward FFT only (no inverse)
- 1D FFTs only (no 2D/3D)
- In-place transforms only
- No multi-GPU support

## License

MIT

## References

- [CUDA cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [Rust FFI Guide](https://doc.rust-lang.org/nomicon/ffi.html)
