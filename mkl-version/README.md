# Batch FFT - Intel MKL Version

C++ implementation using Intel Math Kernel Library (MKL) for high-performance batch FFT processing.

## Requirements

- Intel MKL (Math Kernel Library)
  - Part of Intel oneAPI Base Toolkit
  - Or standalone MKL installation
- CMake 3.10 or later
- C++ compiler with C++11 support

## Installing Intel MKL

### Ubuntu/Debian
```bash
# Add Intel repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

# Install MKL
sudo apt update
sudo apt install intel-oneapi-mkl-devel
```

### Setting up environment
```bash
# For oneAPI installation
source /opt/intel/oneapi/setvars.sh

# Or set MKLROOT manually
export MKLROOT=/opt/intel/oneapi/mkl/latest
```

## Building

```bash
mkdir -p build
cd build
cmake ..
make
```

## Usage

```bash
./batch_fft -b <batch_size> -l <fft_length> -t <threads>
```

### Example

```bash
./batch_fft -b 1000 -l 1024 -t 8
```

## Performance Notes

- Uses MKL's optimized DFT (Discrete Fourier Transform) interface
- Batch processing via `DFTI_NUMBER_OF_TRANSFORMS` configuration
- Fair timing methodology: excludes plan creation from measurements
- Optimized for Intel processors with AVX/AVX2/AVX-512 instructions
