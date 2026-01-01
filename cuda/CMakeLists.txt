cmake_minimum_required(VERSION 3.18)
project(batch_fft_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit (provides cuFFT)
find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(batch_fft src/batch_fft.cu)

# Link cuFFT library
target_link_libraries(batch_fft PRIVATE CUDA::cufft CUDA::cudart)

# Include directories
target_include_directories(batch_fft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Enable optimizations for release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA-specific compiler flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Set CUDA architectures (adjust based on target GPU)
# Default to common architectures, users can override via -DCMAKE_CUDA_ARCHITECTURES
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)  # Pascal, Volta, Turing, Ampere
endif()
