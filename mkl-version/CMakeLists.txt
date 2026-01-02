cmake_minimum_required(VERSION 3.10)
project(batch_fft_mkl)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Intel MKL
# First try using MKLConfig.cmake if available (modern approach)
find_package(MKL CONFIG QUIET)

if(MKL_FOUND)
    message(STATUS "Found MKL via CONFIG")
    add_executable(batch_fft src/batch_fft.cpp)
    target_link_libraries(batch_fft MKL::MKL)
else()
    # Fall back to manual configuration
    message(STATUS "MKL CONFIG not found, trying manual configuration")

    # Try common MKL installation paths
    set(MKL_ROOT $ENV{MKLROOT})
    if(NOT MKL_ROOT)
        if(EXISTS "/opt/intel/oneapi/mkl/latest")
            set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
        elseif(EXISTS "/opt/intel/mkl")
            set(MKL_ROOT "/opt/intel/mkl")
        else()
            message(FATAL_ERROR "Intel MKL not found. Please set MKLROOT environment variable or install Intel MKL.")
        endif()
    endif()

    message(STATUS "Using MKL_ROOT: ${MKL_ROOT}")

    set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")

    # Determine architecture
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(MKL_ARCH "intel64")
    else()
        set(MKL_ARCH "ia32")
    endif()

    set(MKL_LIB_DIR "${MKL_ROOT}/lib/${MKL_ARCH}")

    # MKL libraries for threaded, dynamic linking
    # Using Intel OpenMP for threading
    set(MKL_LIBRARIES
        "${MKL_LIB_DIR}/libmkl_intel_lp64.so"
        "${MKL_LIB_DIR}/libmkl_intel_thread.so"
        "${MKL_LIB_DIR}/libmkl_core.so"
    )

    # Try to find Intel OpenMP library
    # Check both in MKL_ROOT/../compiler and /opt/intel/oneapi/compiler
    set(IOMP_PATHS
        "${MKL_ROOT}/../compiler/latest/lib/libiomp5.so"
        "/opt/intel/oneapi/compiler/latest/lib/libiomp5.so"
        "${MKL_ROOT}/../compiler/latest/lib/intel64/libiomp5.so"
        "${MKL_ROOT}/../compiler/lib/intel64/libiomp5.so"
    )

    set(IOMP_FOUND FALSE)
    foreach(IOMP_PATH ${IOMP_PATHS})
        if(EXISTS "${IOMP_PATH}")
            list(APPEND MKL_LIBRARIES "${IOMP_PATH}")
            message(STATUS "Found Intel OpenMP: ${IOMP_PATH}")
            set(IOMP_FOUND TRUE)
            break()
        endif()
    endforeach()

    if(NOT IOMP_FOUND)
        # Fall back to system pthread
        find_package(Threads REQUIRED)
        list(APPEND MKL_LIBRARIES Threads::Threads)
        message(WARNING "Intel OpenMP not found, using pthreads instead")
    endif()

    add_executable(batch_fft src/batch_fft.cpp)
    target_include_directories(batch_fft PRIVATE ${MKL_INCLUDE_DIR})
    target_link_libraries(batch_fft ${MKL_LIBRARIES} m dl)
endif()

# Enable optimizations for release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
