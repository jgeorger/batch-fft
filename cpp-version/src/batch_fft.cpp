#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fftw3.h>

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

    // Initialize FFTW threading
    fftw_init_threads();
    fftw_plan_with_nthreads(args.threads);

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

    // Create batch FFT plan before timing using FFTW's native batch interface
    // fftw_plan_many_dft parameters:
    //   rank=1: 1D FFT
    //   n: size of each FFT dimension
    //   howmany: number of FFTs to perform (batch size)
    //   in/out: input/output arrays (in-place transform)
    //   inembed/onembed: NULL for simple layout
    //   istride/ostride: 1 (elements are contiguous within each FFT)
    //   idist/odist: length (distance between start of consecutive FFTs)
    int n[] = {static_cast<int>(args.length)};
    fftw_plan plan = fftw_plan_many_dft(
        1,                          // rank (1D)
        n,                          // dimensions of each FFT
        args.batch,                 // number of FFTs (batch size)
        data,                       // input array
        NULL,                       // inembed (NULL = same as n)
        1,                          // istride (elements are contiguous)
        args.length,                // idist (distance between FFTs)
        data,                       // output array (in-place)
        NULL,                       // onembed (NULL = same as n)
        1,                          // ostride
        args.length,                // odist
        FFTW_FORWARD,               // direction
        FFTW_MEASURE                // flags
    );

    // Perform batch FFT with timing
    auto start = std::chrono::high_resolution_clock::now();
    fftw_execute(plan);
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
    fftw_destroy_plan(plan);
    fftw_free(data);
    fftw_cleanup_threads();

    return 0;
}
