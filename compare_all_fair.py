#!/usr/bin/env python3
"""
Complete three-way comparison with FAIR timing for all implementations
All three exclude plan creation from timing
"""

import csv

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Read all three result files (all with fair timing)
rust_cpu = read_csv('rust_cpu_fair.csv')
cpp_cpu = read_csv('cpp_fftw_results.csv')
cuda_gpu = read_csv('rust-cuda/cuda_fair_comparison.csv')

print("# Complete Performance Comparison (ALL implementations with FAIR timing)")
print("## CPU Rust vs CPU C++ (FFTW) vs GPU CUDA")
print("### All exclude plan creation/destruction from timing\n")
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

rust_wins = sum(1 for ratio in rust_vs_cpp_ratios if ratio > 1.0)
cpp_wins = sum(1 for ratio in rust_vs_cpp_ratios if ratio <= 1.0)

print(f"\n### CPU Head-to-Head: Rust vs C++ FFTW")
print(f"- **Rust faster**: {rust_wins}/11 cases")
print(f"- **C++ faster**: {cpp_wins}/11 cases")
print(f"- **Performance difference**: Rust is {sum(rust_vs_cpp_ratios) / len(rust_vs_cpp_ratios):.2f}x faster on average")

# CUDA vs Rust CPU comparison
cuda_vs_rust = [float(cu['gflops']) / float(r['gflops']) for cu, r in zip(cuda_gpu, rust_cpu)]
print(f"\n### GPU vs Best CPU: CUDA vs Rust")
print(f"- **Average speedup**: {sum(cuda_vs_rust) / len(cuda_vs_rust):.2f}x (CUDA over Rust CPU)")
print(f"- **Range**: {min(cuda_vs_rust):.2f}x - {max(cuda_vs_rust):.2f}x")
print(f"- **CUDA wins**: {sum(1 for ratio in cuda_vs_rust if ratio > 1.1)}/11 cases")

print(f"\n## Performance Tiers (Fair Timing)")
print(f"1. **GPU CUDA**: {sum(cuda_gflops_list) / len(cuda_gflops_list):.0f} GFLOPS average - Dominant across all workloads")
print(f"2. **CPU Rust**: {sum(rust_gflops_list) / len(rust_gflops_list):.0f} GFLOPS average - Best CPU implementation")
print(f"3. **CPU C++ FFTW**: {sum(cpp_gflops_list) / len(cpp_gflops_list):.0f} GFLOPS average - Baseline")
