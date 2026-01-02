#!/usr/bin/env python3
"""
Benchmark FFTW implementation with optimal thread count selection
Takes median of 5 runs for each test case
"""

import subprocess
import csv
import sys
import statistics

# Test cases matching previous benchmarks
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

thread_counts = [1, 2, 4, 8]
NUM_RUNS = 5

def run_benchmark(batch, length, threads):
    """Run benchmark NUM_RUNS times and return median result"""
    try:
        times = []
        gflops_values = []

        for _ in range(NUM_RUNS):
            result = subprocess.run(
                ['./build/batch_fft', '-b', str(batch), '-l', str(length), '-t', str(threads)],
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
            times.append(float(data[3]))
            gflops_values.append(float(data[4]))

        if not times:
            return None

        # Return median values
        median_time = statistics.median(times)
        median_gflops = statistics.median(gflops_values)

        return {
            'batch': batch,
            'fft_length': length,
            'threads': threads,
            'time_ms': median_time,
            'gflops': median_gflops
        }
    except subprocess.TimeoutExpired:
        print(f"Timeout for batch={batch}, length={length}, threads={threads}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)
        return None

def find_best_thread_count(batch, length):
    """Find the optimal thread count for a given test case"""
    best_result = None
    best_gflops = 0

    print(f"Testing FFT size={length}, batch={batch}...", file=sys.stderr)

    for threads in thread_counts:
        result = run_benchmark(batch, length, threads)
        if result is None:
            continue

        gflops = result['gflops']
        print(f"  {threads} threads: {gflops:.1f} GFLOPS ({result['time_ms']:.2f} ms)", file=sys.stderr)

        if gflops > best_gflops:
            best_gflops = gflops
            best_result = result

    if best_result:
        print(f"  → Best: {best_result['threads']} threads @ {best_gflops:.1f} GFLOPS\n", file=sys.stderr)

    return best_result

def main():
    print("FFTW Batch FFT Benchmark (Single Precision) - Finding optimal configurations", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results = []

    for batch, length in test_cases:
        result = find_best_thread_count(batch, length)
        if result:
            results.append(result)

    # Write results to CSV
    output_file = 'fftw_results_f32.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['batch', 'fft_length', 'threads', 'time_ms', 'gflops'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_file}", file=sys.stderr)
    print("\nSummary:", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    for r in results:
        size_str = f"{r['fft_length']//1024}K" if r['fft_length'] >= 1024 else str(r['fft_length'])
        print(f"FFT {size_str:>5} × {r['batch']:>5}: {r['threads']}T, {r['time_ms']:>7.2f}ms, {r['gflops']:>4.0f} GFLOPS", file=sys.stderr)

if __name__ == '__main__':
    main()
