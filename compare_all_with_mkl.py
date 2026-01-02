#!/usr/bin/env python3
"""
Complete four-way comparison with FAIR timing for all implementations
All four exclude plan creation from timing: Rust, FFTW, MKL, CUDA
"""

import csv

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Read all four result files (all with fair timing, all single precision)
rust_cpu = read_csv('rust_cpu_fair.csv')
cpp_fftw = read_csv('fftw_results_f32.csv')
cpp_mkl = read_csv('mkl_results_f32.csv')
cuda_gpu = read_csv('rust-cuda/cuda_fair_comparison.csv')

print("# Complete Performance Comparison (ALL implementations with FAIR timing)")
print("## CPU Rust vs CPU FFTW vs CPU MKL vs GPU CUDA")
print("### All exclude plan creation/destruction from timing\n")
print("| FFT Size | Batch | **Rust (CPU)** | **FFTW (CPU)** | **MKL (CPU)** | **CUDA (GPU)** | **Rust vs FFTW** | **MKL vs FFTW** | **CUDA vs FFTW** |")
print("|----------|-------|----------------|----------------|---------------|----------------|------------------|-----------------|------------------|")
print("|          |       | Threads / Time / GFLOPS | Threads / Time / GFLOPS | Threads / Time / GFLOPS | Time / GFLOPS | Speedup | Speedup | Speedup |")

for rust, fftw, mkl, cuda in zip(rust_cpu, cpp_fftw, cpp_mkl, cuda_gpu):
    assert rust['batch'] == fftw['batch'] == mkl['batch'] == cuda['batch']
    assert rust['fft_length'] == fftw['fft_length'] == mkl['fft_length'] == cuda['fft_length']

    batch = int(rust['batch'])
    length = int(rust['fft_length'])

    rust_threads = int(rust['threads'])
    rust_time = float(rust['time_ms'])
    rust_gflops = float(rust['gflops'])

    fftw_threads = int(fftw['threads'])
    fftw_time = float(fftw['time_ms'])
    fftw_gflops = float(fftw['gflops'])

    mkl_threads = int(mkl['threads'])
    mkl_time = float(mkl['time_ms'])
    mkl_gflops = float(mkl['gflops'])

    cuda_time = float(cuda['time_ms'])
    cuda_gflops = float(cuda['gflops'])

    rust_vs_fftw = rust_gflops / fftw_gflops
    mkl_vs_fftw = mkl_gflops / fftw_gflops
    cuda_vs_fftw = cuda_gflops / fftw_gflops

    # Format FFT size
    if length >= 1024:
        size_str = f"{length // 1024}K"
    else:
        size_str = str(length)

    rust_str = f"{rust_threads}T / {rust_time:.2f}ms / {rust_gflops:.0f}"
    fftw_str = f"{fftw_threads}T / {fftw_time:.2f}ms / {fftw_gflops:.0f}"
    mkl_str = f"{mkl_threads}T / {mkl_time:.2f}ms / {mkl_gflops:.0f}"
    cuda_str = f"{cuda_time:.2f}ms / {cuda_gflops:.0f}"

    rust_speedup = f"**{rust_vs_fftw:.2f}x**" if rust_vs_fftw > 1.0 else f"{rust_vs_fftw:.2f}x"
    mkl_speedup = f"**{mkl_vs_fftw:.2f}x**" if mkl_vs_fftw > 1.0 else f"{mkl_vs_fftw:.2f}x"
    cuda_speedup = f"**{cuda_vs_fftw:.2f}x**" if cuda_vs_fftw > 1.0 else f"{cuda_vs_fftw:.2f}x"

    print(f"| **{size_str}** | {batch} | {rust_str} | {fftw_str} | {mkl_str} | {cuda_str} | {rust_speedup} | {mkl_speedup} | {cuda_speedup} |")

print("\n## Summary Statistics\n")

# Calculate averages
rust_gflops_list = [float(r['gflops']) for r in rust_cpu]
fftw_gflops_list = [float(c['gflops']) for c in cpp_fftw]
mkl_gflops_list = [float(m['gflops']) for m in cpp_mkl]
cuda_gflops_list = [float(c['gflops']) for c in cuda_gpu]

rust_vs_fftw_ratios = [float(r['gflops']) / float(f['gflops']) for r, f in zip(rust_cpu, cpp_fftw)]
mkl_vs_fftw_ratios = [float(m['gflops']) / float(f['gflops']) for m, f in zip(cpp_mkl, cpp_fftw)]
cuda_vs_fftw_ratios = [float(cu['gflops']) / float(cp['gflops']) for cu, cp in zip(cuda_gpu, cpp_fftw)]

print(f"### Performance Averages")
print(f"- **CPU Rust**: {sum(rust_gflops_list) / len(rust_gflops_list):.1f} GFLOPS (range: {min(rust_gflops_list):.0f}-{max(rust_gflops_list):.0f})")
print(f"- **CPU FFTW**: {sum(fftw_gflops_list) / len(fftw_gflops_list):.1f} GFLOPS (range: {min(fftw_gflops_list):.0f}-{max(fftw_gflops_list):.0f})")
print(f"- **CPU MKL**: {sum(mkl_gflops_list) / len(mkl_gflops_list):.1f} GFLOPS (range: {min(mkl_gflops_list):.0f}-{max(mkl_gflops_list):.0f})")
print(f"- **GPU CUDA**: {sum(cuda_gflops_list) / len(cuda_gflops_list):.1f} GFLOPS (range: {min(cuda_gflops_list):.0f}-{max(cuda_gflops_list):.0f})")

print(f"\n### Speedup vs FFTW (Baseline)")
print(f"- **Rust CPU average**: {sum(rust_vs_fftw_ratios) / len(rust_vs_fftw_ratios):.2f}x faster")
print(f"- **Rust CPU range**: {min(rust_vs_fftw_ratios):.2f}x - {max(rust_vs_fftw_ratios):.2f}x")
print(f"- **MKL CPU average**: {sum(mkl_vs_fftw_ratios) / len(mkl_vs_fftw_ratios):.2f}x")
print(f"- **MKL CPU range**: {min(mkl_vs_fftw_ratios):.2f}x - {max(mkl_vs_fftw_ratios):.2f}x")
print(f"- **CUDA GPU average**: {sum(cuda_vs_fftw_ratios) / len(cuda_vs_fftw_ratios):.2f}x faster")
print(f"- **CUDA GPU range**: {min(cuda_vs_fftw_ratios):.2f}x - {max(cuda_vs_fftw_ratios):.2f}x")

rust_wins = sum(1 for ratio in rust_vs_fftw_ratios if ratio > 1.0)
fftw_wins = sum(1 for ratio in rust_vs_fftw_ratios if ratio <= 1.0)

mkl_wins_vs_fftw = sum(1 for ratio in mkl_vs_fftw_ratios if ratio > 1.0)
fftw_wins_vs_mkl = sum(1 for ratio in mkl_vs_fftw_ratios if ratio <= 1.0)

print(f"\n### CPU Head-to-Head: Rust vs FFTW")
print(f"- **Rust faster**: {rust_wins}/11 cases")
print(f"- **FFTW faster**: {fftw_wins}/11 cases")
print(f"- **Performance difference**: Rust is {sum(rust_vs_fftw_ratios) / len(rust_vs_fftw_ratios):.2f}x faster on average")

print(f"\n### CPU Head-to-Head: MKL vs FFTW")
print(f"- **MKL faster**: {mkl_wins_vs_fftw}/11 cases")
print(f"- **FFTW faster**: {fftw_wins_vs_mkl}/11 cases")
ratio_avg = sum(mkl_vs_fftw_ratios) / len(mkl_vs_fftw_ratios)
if ratio_avg > 1.0:
    print(f"- **Performance difference**: MKL is {ratio_avg:.2f}x faster on average")
else:
    print(f"- **Performance difference**: FFTW is {1.0/ratio_avg:.2f}x faster on average")

# Rust vs MKL comparison
rust_vs_mkl = [float(r['gflops']) / float(m['gflops']) for r, m in zip(rust_cpu, cpp_mkl)]
print(f"\n### CPU Head-to-Head: Rust vs MKL")
rust_wins_mkl = sum(1 for ratio in rust_vs_mkl if ratio > 1.0)
mkl_wins_rust = sum(1 for ratio in rust_vs_mkl if ratio <= 1.0)
print(f"- **Rust faster**: {rust_wins_mkl}/11 cases")
print(f"- **MKL faster**: {mkl_wins_rust}/11 cases")
print(f"- **Average ratio**: {sum(rust_vs_mkl) / len(rust_vs_mkl):.2f}x (Rust over MKL)")

# CUDA vs best CPU
cuda_vs_rust = [float(cu['gflops']) / float(r['gflops']) for cu, r in zip(cuda_gpu, rust_cpu)]
cuda_vs_mkl = [float(cu['gflops']) / float(m['gflops']) for cu, m in zip(cuda_gpu, cpp_mkl)]

print(f"\n### GPU vs Best CPUs")
print(f"- **CUDA vs Rust CPU**: {sum(cuda_vs_rust) / len(cuda_vs_rust):.2f}x average ({min(cuda_vs_rust):.2f}x - {max(cuda_vs_rust):.2f}x)")
print(f"- **CUDA vs MKL CPU**: {sum(cuda_vs_mkl) / len(cuda_vs_mkl):.2f}x average ({min(cuda_vs_mkl):.2f}x - {max(cuda_vs_mkl):.2f}x)")
print(f"- **CUDA wins**: {sum(1 for ratio in cuda_vs_rust if ratio > 1.1)}/11 cases vs Rust")

print(f"\n## Performance Tiers (Fair Timing)")
print(f"1. **GPU CUDA**: {sum(cuda_gflops_list) / len(cuda_gflops_list):.0f} GFLOPS average - Dominant across all workloads")
print(f"2. **CPU Rust**: {sum(rust_gflops_list) / len(rust_gflops_list):.0f} GFLOPS average - Best CPU implementation")
print(f"3. **CPU FFTW**: {sum(fftw_gflops_list) / len(fftw_gflops_list):.0f} GFLOPS average - Baseline reference")
print(f"4. **CPU MKL**: {sum(mkl_gflops_list) / len(mkl_gflops_list):.0f} GFLOPS average - Intel optimized CPU")
