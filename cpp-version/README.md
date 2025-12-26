# Batch FFT - C++ Implementation

C++ implementation of the batch 1D Complex-to-Complex (C2C) Fast Fourier Transform processor for performance comparison with the Rust version.

## Features

- **Batch Processing**: Process multiple FFTs in parallel from a contiguous array
- **Multi-threaded**: Uses OpenMP for parallel processing
- **Performance Metrics**: Real-time GFLOPS calculation and timing
- **CSV Output**: Same output format as Rust version for easy comparison

## Requirements

- C++11 or later
- CMake 3.10 or later
- FFTW3 library
- OpenMP support

## Installation

### Install Dependencies (macOS)

```bash
brew install fftw cmake
```

### Install Dependencies (Ubuntu/Debian)

```bash
sudo apt-get install libfftw3-dev cmake libomp-dev
```

### Build

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Usage

```bash
./batch_fft -b <BATCH_SIZE> -l <FFT_LENGTH> -t <NUM_THREADS>
```

### Arguments

- `-b, --batch`: Number of FFTs in the batch
- `-l, --length`: FFT transform length (number of samples per FFT)
- `-t, --threads`: Number of threads to use for parallel processing

### Example

Process 1000 FFTs of length 1024 using 8 threads:

```bash
./batch_fft -b 1000 -l 1024 -t 8
```

## Output

CSV format with header and data:

```
batch,fft_length,threads,time_ms,gflops
1000,1024,8,1.234,42
```

## Performance Notes

- Uses FFTW3 for high-performance FFT computation
- OpenMP provides parallel execution across batch elements
- FLOPS calculation: `Batch × 5 × N × log2(N)` total FLOPs
- Compiled with `-O3 -march=native` for optimal performance
