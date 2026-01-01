#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include "cuda_helpers.h"

struct Args {
    size_t batch;
    size_t length;
};

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " -b <batch> -l <length>\n";
    std::cerr << "  -b, --batch    Number of FFTs in the batch\n";
    std::cerr << "  -l, --length   FFT transform length\n";
}

bool parse_args(int argc, char* argv[], Args& args) {
    args.batch = 0;
    args.length = 0;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch") == 0) && i + 1 < argc) {
            args.batch = std::stoull(argv[++i]);
        } else if ((strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--length") == 0) && i + 1 < argc) {
            args.length = std::stoull(argv[++i]);
        } else {
            return false;
        }
    }

    return args.batch > 0 && args.length > 0;
}

void generate_input_data(cufftComplex* h_data, size_t batch, size_t length) {
    size_t total_size = batch * length;

    // Generate sample data (sine wave with varying frequencies)
    for (size_t i = 0; i < total_size; i++) {
        double t = static_cast<double>(i % length) / static_cast<double>(length);
        size_t batch_idx = i / length;
        double freq = 1.0 + static_cast<double>(batch_idx);
        h_data[i].x = std::cos(2.0 * M_PI * freq * t);  // Real part
        h_data[i].y = 0.0;  // Imaginary part
    }
}

void perform_batch_fft(cufftComplex* d_data, size_t batch, size_t length) {
    cufftHandle plan;
    int n[] = {static_cast<int>(length)};

    // Create batch plan for all FFTs
    // Parameters match FFTW's fftw_plan_many_dft
    CUFFT_CHECK(cufftPlanMany(
        &plan,
        1,                          // rank (1D)
        n,                          // dimensions of each FFT
        NULL,                       // inembed (NULL = same as n)
        1,                          // istride (elements are contiguous)
        length,                     // idist (distance between FFTs)
        NULL,                       // onembed (NULL = same as n)
        1,                          // ostride
        length,                     // odist
        CUFFT_C2C,                 // type (complex-to-complex)
        batch                       // batch size
    ));

    // Execute batch FFT (cuFFT handles GPU parallelism internally)
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    // Synchronize to ensure completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    CUFFT_CHECK(cufftDestroy(plan));
}

double calculate_flops(size_t batch, size_t length) {
    double n = static_cast<double>(length);
    double b = static_cast<double>(batch);
    return b * 5.0 * n * std::log2(n);
}

int main(int argc, char* argv[]) {
    Args args;

    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    // Allocate host memory and generate data
    size_t total_size = args.batch * args.length;
    cufftComplex* h_data = new cufftComplex[total_size];
    generate_input_data(h_data, args.batch, args.length);

    // Allocate device memory
    cufftComplex* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(cufftComplex) * total_size));

    // Copy data to device (NOT timed)
    CUDA_CHECK(cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * total_size,
                          cudaMemcpyHostToDevice));

    // Warm-up run for accurate timing
    perform_batch_fft(d_data, args.batch, args.length);

    // Re-copy data for actual timed run
    CUDA_CHECK(cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * total_size,
                          cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Perform timed FFT
    CUDA_CHECK(cudaEventRecord(start));
    perform_batch_fft(d_data, args.batch, args.length);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Calculate performance metrics
    double flops = calculate_flops(args.batch, args.length);
    double gflops = flops / (milliseconds / 1000.0) / 1e9;

    // Output results as CSV
    std::cout << "batch,fft_length,time_ms,gflops\n";
    std::cout << args.batch << "," << args.length << ","
              << std::fixed << std::setprecision(3) << milliseconds << ","
              << std::fixed << std::setprecision(0) << gflops << "\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;

    return 0;
}
