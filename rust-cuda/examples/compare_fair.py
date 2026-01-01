#!/usr/bin/env python3
"""
Fair comparison: Rust FFI vs CUDA C++ (both excluding plan creation)
"""

import csv

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Read both result files
cuda_results = read_csv('../cuda_fair_comparison.csv')
rust_results = read_csv('rust_cuda_results.csv')

# Create comparison table
print("# Fair Performance Comparison: Rust FFI vs CUDA C++")
print("## (Both exclude plan creation from timing)\n")
print("| FFT Size | Batch | CUDA C++ | Rust FFI | Ratio |")
print("|----------|-------|----------|----------|-------|")
print("|          |       | Time / GFLOPS | Time / GFLOPS | C++/Rust |")

cuda_faster = 0
rust_faster = 0

for cuda, rust in zip(cuda_results, rust_results):
    assert cuda['batch'] == rust['batch']
    assert cuda['fft_length'] == rust['fft_length']

    batch = int(cuda['batch'])
    length = int(cuda['fft_length'])

    cuda_time = float(cuda['time_ms'])
    cuda_gflops = float(cuda['gflops'])
    rust_time = float(rust['time_ms'])
    rust_gflops = float(rust['gflops'])

    ratio = cuda_time / rust_time

    if ratio > 1.0:
        rust_faster += 1
        ratio_str = f"{ratio:.2f}x (Rust faster)"
    elif ratio < 1.0:
        cuda_faster += 1
        ratio_str = f"{1/ratio:.2f}x (C++ faster)"
    else:
        ratio_str = "~1.00x (tie)"

    # Format FFT size
    if length >= 1024:
        size_str = f"{length // 1024}K"
    else:
        size_str = str(length)

    print(f"| **{size_str}** | {batch} | "
          f"{cuda_time:.3f}ms / {cuda_gflops:.0f} | "
          f"{rust_time:.3f}ms / {rust_gflops:.0f} | "
          f"{ratio_str} |")

print("\n## Summary Statistics\n")

# Calculate statistics
time_ratios = [float(c['time_ms']) / float(r['time_ms'])
               for c, r in zip(cuda_results, rust_results)]
avg_ratio = sum(time_ratios) / len(time_ratios)

gflops_ratios = [float(c['gflops']) / float(r['gflops'])
                 for c, r in zip(cuda_results, rust_results)]
avg_gflops_ratio = sum(gflops_ratios) / len(gflops_ratios)

cuda_avg_gflops = sum(float(c['gflops']) for c in cuda_results) / len(cuda_results)
rust_avg_gflops = sum(float(r['gflops']) for r in rust_results) / len(rust_results)

print(f"- **Test Cases**: {len(cuda_results)} configurations")
print(f"- **CUDA C++ faster**: {cuda_faster} cases")
print(f"- **Rust FFI faster**: {rust_faster} cases")
print(f"- **Average time ratio**: {avg_ratio:.2f}x (C++/Rust)")
print(f"- **Average GFLOPS ratio**: {avg_gflops_ratio:.2f}x (C++/Rust)")
print(f"- **CUDA C++ average performance**: {cuda_avg_gflops:.0f} GFLOPS")
print(f"- **Rust FFI average performance**: {rust_avg_gflops:.0f} GFLOPS")
print(f"- **Peak CUDA C++ performance**: {max(float(c['gflops']) for c in cuda_results):.0f} GFLOPS")
print(f"- **Peak Rust FFI performance**: {max(float(r['gflops']) for r in rust_results):.0f} GFLOPS")

# Performance difference percentage
perf_diff_pct = ((avg_gflops_ratio - 1.0) * 100)
if perf_diff_pct > 0:
    print(f"- **Overall**: CUDA C++ is {perf_diff_pct:.1f}% faster on average")
else:
    print(f"- **Overall**: Rust FFI is {-perf_diff_pct:.1f}% faster on average")

# Detailed CSV output
print("\n## Detailed Results\n")
print("```csv")
print("fft_size,batch,cuda_time_ms,cuda_gflops,rust_time_ms,rust_gflops,time_ratio,gflops_ratio")
for cuda, rust in zip(cuda_results, rust_results):
    time_ratio = float(cuda['time_ms']) / float(rust['time_ms'])
    gflops_ratio = float(cuda['gflops']) / float(rust['gflops'])
    print(f"{cuda['fft_length']},{cuda['batch']},"
          f"{cuda['time_ms']},{cuda['gflops']},"
          f"{rust['time_ms']},{rust['gflops']},"
          f"{time_ratio:.3f},{gflops_ratio:.3f}")
print("```")

print("\n## Conclusion\n")
if abs(perf_diff_pct) < 10:
    print("✅ **Performance parity achieved!**")
    print(f"   Both implementations perform within {abs(perf_diff_pct):.1f}% of each other.")
    print("   The minor differences are likely due to:")
    print("   - FFI overhead in Rust (function call boundary)")
    print("   - Compiler optimizations")
    print("   - Measurement variance")
    print("\n   Both implementations effectively utilize the same cuFFT library.")
else:
    print(f"⚠️  Performance difference: {abs(perf_diff_pct):.1f}%")
    print("   This warrants further investigation.")
