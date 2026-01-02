#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mkl.h>
#include <mkl_dfti.h>

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

    // Set MKL thread count
    mkl_set_num_threads(args.threads);

    // Initialize input data: batch of signals in a contiguous array
    size_t total_size = args.batch * args.length;
    std::vector<std::complex<float>> data(total_size);

    // Generate sample data (sine wave with varying frequencies)
    for (size_t i = 0; i < total_size; i++) {
        float t = static_cast<float>(i % args.length) / static_cast<float>(args.length);
        size_t batch_idx = i / args.length;
        float freq = 1.0f + static_cast<float>(batch_idx);
        data[i] = std::complex<float>(std::cos(2.0f * M_PI * freq * t), 0.0f);
    }

    // Create MKL FFT descriptor for batch processing (single precision)
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;
    MKL_LONG status;

    // Create descriptor for 1D complex-to-complex FFT (single precision)
    status = DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, args.length);
    if (status != DFTI_NO_ERROR) {
        std::cerr << "Error creating MKL descriptor: " << DftiErrorMessage(status) << std::endl;
        return 1;
    }

    // Configure for batch processing
    status = DftiSetValue(handle, DFTI_NUMBER_OF_TRANSFORMS, args.batch);
    if (status != DFTI_NO_ERROR) {
        std::cerr << "Error setting number of transforms: " << DftiErrorMessage(status) << std::endl;
        DftiFreeDescriptor(&handle);
        return 1;
    }

    // Set input stride (distance between consecutive elements in same transform)
    status = DftiSetValue(handle, DFTI_INPUT_DISTANCE, args.length);
    if (status != DFTI_NO_ERROR) {
        std::cerr << "Error setting input distance: " << DftiErrorMessage(status) << std::endl;
        DftiFreeDescriptor(&handle);
        return 1;
    }

    // Set output stride (distance between consecutive elements in same transform)
    status = DftiSetValue(handle, DFTI_OUTPUT_DISTANCE, args.length);
    if (status != DFTI_NO_ERROR) {
        std::cerr << "Error setting output distance: " << DftiErrorMessage(status) << std::endl;
        DftiFreeDescriptor(&handle);
        return 1;
    }

    // In-place transform
    status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_INPLACE);
    if (status != DFTI_NO_ERROR) {
        std::cerr << "Error setting placement: " << DftiErrorMessage(status) << std::endl;
        DftiFreeDescriptor(&handle);
        return 1;
    }

    // Commit the descriptor (creates the plan - done before timing)
    status = DftiCommitDescriptor(handle);
    if (status != DFTI_NO_ERROR) {
        std::cerr << "Error committing descriptor: " << DftiErrorMessage(status) << std::endl;
        DftiFreeDescriptor(&handle);
        return 1;
    }

    // Perform batch FFT with timing (fair timing - excludes plan creation)
    auto start = std::chrono::high_resolution_clock::now();
    status = DftiComputeForward(handle, data.data());
    auto end = std::chrono::high_resolution_clock::now();

    if (status != DFTI_NO_ERROR) {
        std::cerr << "Error computing FFT: " << DftiErrorMessage(status) << std::endl;
        DftiFreeDescriptor(&handle);
        return 1;
    }

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
    DftiFreeDescriptor(&handle);

    return 0;
}
