#!/bin/bash
# Benchmark script for CUDA Batch FFT

set -e

# Check if batch_fft executable exists
if [ ! -f "../build/batch_fft" ]; then
    echo "Error: batch_fft executable not found"
    echo "Please build the project first:"
    echo "  cd ../build && cmake .. && make"
    exit 1
fi

echo "===================================="
echo "  CUDA Batch FFT Benchmark Suite"
echo "===================================="
echo ""

# Test different batch sizes with fixed FFT length
echo "Test 1: Varying Batch Sizes (FFT length: 1024)"
echo "------------------------------------------------"
echo ""

for batch in 100 500 1000 2000; do
    echo "Batch size: $batch"
    ../build/batch_fft -b $batch -l 1024 | tail -1
    echo ""
done

echo ""
echo "Test 2: Varying FFT Lengths (Batch size: 1000)"
echo "------------------------------------------------"
echo ""

for length in 256 512 1024 2048 4096 8192; do
    echo "FFT length: $length"
    ../build/batch_fft -b 1000 -l $length | tail -1
    echo ""
done

echo ""
echo "Test 3: Large FFT Workloads"
echo "------------------------------------------------"
echo ""

configs=(
    "500 8192"
    "1000 4096"
    "2000 2048"
    "250 16384"
)

for config in "${configs[@]}"; do
    read -r batch length <<< "$config"
    echo "Batch: $batch, FFT length: $length"
    ../build/batch_fft -b $batch -l $length | tail -1
    echo ""
done

echo ""
echo "===================================="
echo "  Benchmark Complete"
echo "===================================="
