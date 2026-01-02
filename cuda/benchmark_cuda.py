#!/usr/bin/env python3
"""
Benchmark pure CUDA C++ implementation
Takes median of 5 runs for each test case
"""

import subprocess
import csv
import sys
import statistics

# Test cases matching other benchmarks
test_cases = [
    (1000, 1024),   # 1K FFT, batch 1000
    (10000, 1024),  # 1K FFT, batch 10000
    (1000, 2048),   # 2K FFT
    (1000, 4096),   # 4K FFT
    (500, 8192),    # 8K FFT
    (500, 16384),   # 16K FFT
    (250, 32768),   # 32K FFT
    (250, 65536),   # 64K FFT
    (250, 131072),  # 128K FFT
    (250, 262144),  # 256K FFT
    (250, 524288),  # 512K FFT
]

NUM_RUNS = 5

def run_benchmark(batch, length):
    """Run benchmark NUM_RUNS times and return median result"""
    try:
        times = []
        gflops_values = []

        for _ in range(NUM_RUNS):
            result = subprocess.run(
                ['./build/batch_fft', '-b', str(batch), '-l', str(length)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                print(f"Error running benchmark: {result.stderr}", file=sys.stderr)
                continue

            # Parse CSV output (skip header)
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                continue

            data = lines[1].split(',')
            times.append(float(data[2]))
            gflops_values.append(float(data[3]))

        if not times:
            return None

        # Return median values
        median_time = statistics.median(times)
        median_gflops = statistics.median(gflops_values)

        return {
            'batch': batch,
            'fft_length': length,
            'time_ms': median_time,
            'gflops': median_gflops
        }
    except subprocess.TimeoutExpired:
        print(f"Timeout for batch={batch}, length={length}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)
        return None

def main():
    print("CUDA C++ Batch FFT Benchmark", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results = []

    for batch, length in test_cases:
        print(f"Testing FFT size={length}, batch={batch}...", file=sys.stderr)
        result = run_benchmark(batch, length)
        if result:
            gflops = result['gflops']
            print(f"  Result: {result['time_ms']:.2f}ms, {gflops:.0f} GFLOPS\n", file=sys.stderr)
            results.append(result)

    # Write results to CSV
    output_file = 'cuda_cpp_results.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['batch', 'fft_length', 'time_ms', 'gflops'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_file}", file=sys.stderr)
    print("\nSummary:", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    for r in results:
        size_str = f"{r['fft_length']//1024}K" if r['fft_length'] >= 1024 else str(r['fft_length'])
        print(f"FFT {size_str:>5} × {r['batch']:>5}: {r['time_ms']:>7.2f}ms, {r['gflops']:>4.0f} GFLOPS", file=sys.stderr)

if __name__ == '__main__':
    main()
