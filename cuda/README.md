# Batch FFT - CUDA Implementation

CUDA/cuFFT implementation of the batch 1D Complex-to-Complex (C2C) Fast Fourier Transform processor for GPU acceleration and performance comparison with CPU versions.

## Features

- **GPU Acceleration**: Uses NVIDIA cuFFT library for high-performance FFT computation
- **Batch Processing**: Leverages cuFFT's native batch API (`cufftPlanMany`)
- **Automatic Parallelization**: cuFFT optimally utilizes all GPU resources
- **Performance Metrics**: Real-time GFLOPS calculation and timing
- **CSV Output**: Compatible output format for comparison with CPU versions

## Requirements

- CUDA Toolkit 11.0 or later (tested with 12.x)
- CMake 3.18 or later
- NVIDIA GPU with compute capability 6.0+ (Pascal or newer)
- C++11 compatible compiler

## Installation

### Install Dependencies (Ubuntu/Debian)

```bash
# Install CUDA Toolkit (if not already installed)
# Follow NVIDIA's official installation guide for your distribution
# https://developer.nvidia.com/cuda-downloads

sudo apt-get install cmake
```

### Build

```bash
cd cuda
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

**Note**: By default, the code is compiled for multiple GPU architectures (60, 70, 75, 80, 86). To target a specific architecture:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 ..
```

Common compute capabilities:
- **60**: Pascal (GTX 10 series, Tesla P100)
- **70**: Volta (Tesla V100)
- **75**: Turing (RTX 20 series, GTX 16 series)
- **80**: Ampere (A100)
- **86**: Ampere (RTX 30 series)

## Usage

```bash
./batch_fft -b <BATCH_SIZE> -l <FFT_LENGTH>
```

### Arguments

- `-b, --batch`: Number of FFTs in the batch
- `-l, --length`: FFT transform length (number of samples per FFT)

**Note**: Unlike the CPU versions, there is no threads parameter. cuFFT automatically optimizes GPU resource utilization.

### Examples

Process 1000 FFTs of length 1024:

```bash
./batch_fft -b 1000 -l 1024
```

Process 2000 FFTs of length 2048:

```bash
./batch_fft -b 2000 -l 2048
```

Large batch of 8k FFTs:

```bash
./batch_fft -b 500 -l 8192
```

## Output

CSV format with header and data:

```
batch,fft_length,time_ms,gflops
1000,1024,0.523,98
```

**CSV Fields:**
- `batch`: Number of FFTs in the batch
- `fft_length`: FFT transform length
- `time_ms`: Execution time in milliseconds (GPU FFT only, excludes data transfer)
- `gflops`: Performance in GFLOPS (billions of FLOPs per second)

## Implementation Details

### cuFFT Batch Processing

The implementation uses cuFFT's `cufftPlanMany()` API for efficient batch FFT processing:

```cpp
cufftPlanMany(&plan,
              1,           // rank (1D)
              n,           // dimensions
              NULL,        // inembed
              1,           // istride (contiguous)
              length,      // idist (distance between FFTs)
              NULL,        // onembed
              1,           // ostride
              length,      // odist
              CUFFT_C2C,   // type (complex-to-complex)
              batch);      // batch size
```

This single call processes all FFTs in the batch, with cuFFT handling GPU parallelism internally.

### Memory Management

- Input data generated on CPU (host)
- Explicit GPU memory allocation with `cudaMalloc`
- Data transfer to GPU before timing
- In-place FFT execution (input and output use same buffer)
- Timing uses CUDA events for precise GPU-only measurement

### Timing Methodology

1. **Warm-up run**: First FFT execution (excluded from timing)
2. **Data re-copy**: Reset data to initial state
3. **Timed run**: Second FFT execution with CUDA event timing
4. **Synchronization**: Ensure all GPU work completes

This approach ensures accurate timing that excludes:
- Data generation
- Host-to-device memory transfer
- cuFFT plan creation
- First-run initialization overhead

### Performance Metrics

- **FLOPS calculation**: `Batch × 5 × N × log2(N)` (same as CPU versions)
- **GFLOPS**: FLOPS / (execution_time_seconds × 10^9)

## Performance Notes

- Typical speedup over CPU: **2-100x** depending on FFT size and batch size
- Best performance with:
  - Large batch sizes (>= 500)
  - Medium to large FFT lengths (>= 1024)
  - Power-of-2 FFT lengths (optimal for FFT algorithms)
- Expected GFLOPS: **500-2000** on modern GPUs (vs 60-79 on CPU)
- Compilation with multiple architectures increases binary size but ensures portability

## Troubleshooting

### No CUDA-capable device

```
CUDA Error: no CUDA-capable device is detected
```

**Solution**: Ensure you have an NVIDIA GPU and the CUDA driver installed:

```bash
nvidia-smi  # Check if GPU is detected
```

### Out of memory

```
CUDA Error: out of memory
```

**Solution**: Reduce batch size or FFT length. GPU memory required:
```
Memory = batch × length × sizeof(cufftComplex)
       = batch × length × 8 bytes
```

For example:
- 1000 × 1024: ~8 MB
- 2000 × 2048: ~32 MB
- 1000 × 8192: ~64 MB

### Architecture mismatch

If you see warnings about GPU architecture or poor performance:

```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Rebuild for your specific architecture (e.g., 8.6 for RTX 3090)
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
make
```

### cuFFT errors

```
cuFFT Error: CUFFT_INVALID_SIZE
```

**Solution**: Ensure FFT length is valid. cuFFT supports most sizes, but power-of-2 sizes (256, 512, 1024, 2048, etc.) are optimal.

## Performance Comparison with CPU Versions

Compare with Rust and C++ versions:

```bash
# CUDA version (from cuda/build/)
./batch_fft -b 1000 -l 1024

# Rust version (from project root)
cargo run --release -- -b 1000 -l 1024 -t 8

# C++ version (from cpp-version/build/)
./batch_fft -b 1000 -l 1024 -t 8
```

All three versions output the same CSV format for easy comparison.

### Example Comparison

Workload: 1000 × 1024 FFTs

| Implementation | Time (ms) | GFLOPS | Speedup |
|----------------|-----------|--------|---------|
| Rust (8 threads) | 1.50 | 34 | 1.0x (baseline) |
| C++ (8 threads) | 1.74 | 29 | 0.9x |
| **CUDA** | **0.52** | **98** | **2.9x** |

*Note: Actual performance varies by hardware. GPU results are on mid-range GPU; high-end GPUs achieve higher speedups.*

## License

Same as parent project (MIT)
