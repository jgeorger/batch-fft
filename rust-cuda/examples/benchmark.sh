#!/bin/bash
# Comprehensive benchmark for Rust FFI cuFFT implementation
# Matches all test configurations from README benchmark table

set -e

BINARY="../target/release/batch_fft"

if [ ! -f "$BINARY" ]; then
    echo "Building release binary..."
    cd .. && ~/.cargo/bin/cargo build --release
    cd examples
fi

echo "batch,fft_length,time_ms,gflops" > rust_cuda_results.csv

# All 11 test configurations matching README
declare -a tests=(
    "1000 1024"
    "10000 1024"
    "1000 2048"
    "1000 4096"
    "500 8192"
    "500 16384"
    "250 32768"
    "250 65536"
    "250 131072"
    "250 262144"
    "250 524288"
)

echo "Running Rust FFI cuFFT benchmarks..."
for test in "${tests[@]}"; do
    read -r batch length <<< "$test"
    echo "  batch=$batch, length=$length"
    $BINARY -b $batch -l $length | tail -1 >> rust_cuda_results.csv
done

echo ""
echo "Results written to rust_cuda_results.csv"
echo ""
cat rust_cuda_results.csv
