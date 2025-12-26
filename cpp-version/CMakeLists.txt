cmake_minimum_required(VERSION 3.10)
project(batch_fft_cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# macOS specific: Use Homebrew's libomp
if(APPLE)
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

# Find required packages
find_package(OpenMP REQUIRED)

# Find FFTW3
if(APPLE)
    execute_process(COMMAND brew --prefix fftw
                    OUTPUT_VARIABLE FFTW_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(FFTW_PREFIX)
        set(FFTW_INCLUDE_DIRS "${FFTW_PREFIX}/include")
        set(FFTW_LIBRARIES "${FFTW_PREFIX}/lib/libfftw3.dylib")
    endif()
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW REQUIRED fftw3)
endif()

# Add executable
add_executable(batch_fft src/batch_fft.cpp)

# Set compile options for OpenMP
if(APPLE AND LIBOMP_PREFIX)
    target_compile_options(batch_fft PRIVATE -Xpreprocessor -fopenmp)
    target_include_directories(batch_fft PRIVATE ${LIBOMP_PREFIX}/include)
    target_link_directories(batch_fft PRIVATE ${LIBOMP_PREFIX}/lib)
endif()

# Link libraries
target_link_libraries(batch_fft
    OpenMP::OpenMP_CXX
    ${FFTW_LIBRARIES}
)

target_include_directories(batch_fft PRIVATE ${FFTW_INCLUDE_DIRS})

# Enable optimizations for release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
