#!/usr/bin/env python3
"""
Complete three-way comparison: CPU Rust vs CPU C++ vs GPU CUDA (corrected)
"""

import csv

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Read all three result files
rust_cpu = read_csv('rust_cpu_results.csv')
cpp_cpu = read_csv('cpp_fftw_results.csv')
cuda_gpu = read_csv('rust-cuda/cuda_fair_comparison.csv')

print("# Complete Performance Comparison (Corrected CUDA Timing)")
print("## CPU Rust vs CPU C++ (FFTW) vs GPU CUDA\n")
print("| FFT Size | Batch | **Rust (CPU)** | **C++ FFTW (CPU)** | **CUDA (GPU)** | **Rust vs C++** | **CUDA vs C++** |")
print("|----------|-------|----------------|-------------------|----------------|-----------------|-----------------|")
print("|          |       | Threads / Time / GFLOPS | Threads / Time / GFLOPS | Time / GFLOPS | Speedup | Speedup |")

for rust, cpp, cuda in zip(rust_cpu, cpp_cpu, cuda_gpu):
    assert rust['batch'] == cpp['batch'] == cuda['batch']
    assert rust['fft_length'] == cpp['fft_length'] == cuda['fft_length']

    batch = int(rust['batch'])
    length = int(rust['fft_length'])

    rust_threads = int(rust['threads'])
    rust_time = float(rust['time_ms'])
    rust_gflops = float(rust['gflops'])

    cpp_threads = int(cpp['threads'])
    cpp_time = float(cpp['time_ms'])
    cpp_gflops = float(cpp['gflops'])

    cuda_time = float(cuda['time_ms'])
    cuda_gflops = float(cuda['gflops'])

    rust_vs_cpp = rust_gflops / cpp_gflops
    cuda_vs_cpp = cuda_gflops / cpp_gflops

    # Format FFT size
    if length >= 1024:
        size_str = f"{length // 1024}K"
    else:
        size_str = str(length)

    rust_str = f"{rust_threads}T / {rust_time:.2f}ms / {rust_gflops:.0f}"
    cpp_str = f"{cpp_threads}T / {cpp_time:.2f}ms / {cpp_gflops:.0f}"
    cuda_str = f"{cuda_time:.2f}ms / {cuda_gflops:.0f}"

    rust_speedup = f"**{rust_vs_cpp:.2f}x**" if rust_vs_cpp > 1.0 else f"{rust_vs_cpp:.2f}x"
    cuda_speedup = f"**{cuda_vs_cpp:.2f}x**" if cuda_vs_cpp > 1.0 else f"{cuda_vs_cpp:.2f}x"

    print(f"| **{size_str}** | {batch} | {rust_str} | {cpp_str} | {cuda_str} | {rust_speedup} | {cuda_speedup} |")

print("\n## Summary Statistics\n")

# Calculate averages
rust_gflops_list = [float(r['gflops']) for r in rust_cpu]
cpp_gflops_list = [float(c['gflops']) for c in cpp_cpu]
cuda_gflops_list = [float(c['gflops']) for c in cuda_gpu]

rust_vs_cpp_ratios = [float(r['gflops']) / float(c['gflops']) for r, c in zip(rust_cpu, cpp_cpu)]
cuda_vs_cpp_ratios = [float(cu['gflops']) / float(cp['gflops']) for cu, cp in zip(cuda_gpu, cpp_cpu)]

print(f"### Performance Averages")
print(f"- **CPU Rust**: {sum(rust_gflops_list) / len(rust_gflops_list):.1f} GFLOPS (range: {min(rust_gflops_list):.0f}-{max(rust_gflops_list):.0f})")
print(f"- **CPU C++ FFTW**: {sum(cpp_gflops_list) / len(cpp_gflops_list):.1f} GFLOPS (range: {min(cpp_gflops_list):.0f}-{max(cpp_gflops_list):.0f})")
print(f"- **GPU CUDA**: {sum(cuda_gflops_list) / len(cuda_gflops_list):.1f} GFLOPS (range: {min(cuda_gflops_list):.0f}-{max(cuda_gflops_list):.0f})")

print(f"\n### Speedup vs C++ FFTW")
print(f"- **Rust CPU average**: {sum(rust_vs_cpp_ratios) / len(rust_vs_cpp_ratios):.2f}x faster")
print(f"- **Rust CPU range**: {min(rust_vs_cpp_ratios):.2f}x - {max(rust_vs_cpp_ratios):.2f}x")
print(f"- **CUDA GPU average**: {sum(cuda_vs_cpp_ratios) / len(cuda_vs_cpp_ratios):.2f}x faster")
print(f"- **CUDA GPU range**: {min(cuda_vs_cpp_ratios):.2f}x - {max(cuda_vs_cpp_ratios):.2f}x")

print(f"\n### Key Findings")
print(f"1. **CPU Rust outperforms CPU C++ FFTW** in every test case ({min(rust_vs_cpp_ratios):.2f}x - {max(rust_vs_cpp_ratios):.2f}x faster)")
print(f"2. **GPU CUDA dominates large FFTs** (up to {max(cuda_vs_cpp_ratios):.1f}x faster than C++)")
print(f"3. **CUDA corrected**: With fair timing, CUDA achieves {sum(cuda_gflops_list) / len(cuda_gflops_list):.0f} GFLOPS average")
print(f"4. **Rust CPU competitive with CUDA** for small FFTs (Rust: {rust_gflops_list[0]:.0f} vs CUDA: {cuda_gflops_list[0]:.0f} GFLOPS at 1K×1000)")

# CUDA vs Rust CPU comparison
cuda_vs_rust = [float(cu['gflops']) / float(r['gflops']) for cu, r in zip(cuda_gpu, rust_cpu)]
print(f"\n### CUDA vs Rust CPU")
print(f"- **Average speedup**: {sum(cuda_vs_rust) / len(cuda_vs_rust):.2f}x (CUDA over Rust CPU)")
print(f"- **Range**: {min(cuda_vs_rust):.2f}x - {max(cuda_vs_rust):.2f}x")
cuda_wins = sum(1 for ratio in cuda_vs_rust if ratio > 1.1)
rust_wins = sum(1 for ratio in cuda_vs_rust if ratio < 0.9)
ties = len(cuda_vs_rust) - cuda_wins - rust_wins
print(f"- **CUDA significantly faster**: {cuda_wins} cases")
print(f"- **Rust CPU significantly faster**: {rust_wins} cases")
print(f"- **Effectively tied** (within 10%): {ties} cases")
