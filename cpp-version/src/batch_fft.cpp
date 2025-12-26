#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fftw3.h>
#include <omp.h>

struct Args {
    size_t batch;
    size_t length;
    int threads;
};

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " -b <batch> -l <length> -t <threads>\n";
    std::cerr << "  -b, --batch    Number of FFTs in the batch\n";
    std::cerr << "  -l, --length   FFT transform length\n";
    std::cerr << "  -t, --threads  Number of threads to use\n";
}

bool parse_args(int argc, char* argv[], Args& args) {
    args.batch = 0;
    args.length = 0;
    args.threads = 0;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch") == 0) && i + 1 < argc) {
            args.batch = std::stoull(argv[++i]);
        } else if ((strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--length") == 0) && i + 1 < argc) {
            args.length = std::stoull(argv[++i]);
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
            args.threads = std::stoi(argv[++i]);
        } else {
            return false;
        }
    }

    return args.batch > 0 && args.length > 0 && args.threads > 0;
}

void perform_batch_fft(fftw_complex* data, size_t batch, size_t length, int threads) {
    // Create a single plan to use as a template
    fftw_complex* temp = fftw_alloc_complex(length);
    fftw_plan plan = fftw_plan_dft_1d(length, temp, temp, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_free(temp);

    // Process batches in parallel using the same plan
    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < batch; i++) {
        fftw_complex* signal = &data[i * length];
        // Execute FFT on this batch element using new-array execution
        fftw_execute_dft(plan, signal, signal);
    }

    fftw_destroy_plan(plan);
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

    // Initialize input data: batch of signals in a contiguous array
    size_t total_size = args.batch * args.length;
    fftw_complex* data = fftw_alloc_complex(total_size);

    // Generate sample data (sine wave with varying frequencies)
    for (size_t i = 0; i < total_size; i++) {
        double t = static_cast<double>(i % args.length) / static_cast<double>(args.length);
        size_t batch_idx = i / args.length;
        double freq = 1.0 + static_cast<double>(batch_idx);
        data[i][0] = std::cos(2.0 * M_PI * freq * t);  // Real part
        data[i][1] = 0.0;  // Imaginary part
    }

    // Perform batch FFT with timing
    auto start = std::chrono::high_resolution_clock::now();
    perform_batch_fft(data, args.batch, args.length, args.threads);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate performance metrics
    std::chrono::duration<double> duration = end - start;
    double time_ms = duration.count() * 1000.0;
    double flops = calculate_flops(args.batch, args.length);
    double gflops = flops / duration.count() / 1e9;

    // Output results as CSV
    std::cout << "batch,fft_length,threads,time_ms,gflops\n";
    std::cout << args.batch << "," << args.length << "," << args.threads << ","
              << std::fixed << std::setprecision(3) << time_ms << ","
              << std::fixed << std::setprecision(0) << gflops << "\n";

    // Cleanup
    fftw_free(data);

    return 0;
}
