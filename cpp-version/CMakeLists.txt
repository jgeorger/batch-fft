cmake_minimum_required(VERSION 3.10)
project(batch_fft_cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find FFTW3 and FFTW3 threads
if(APPLE)
    execute_process(COMMAND brew --prefix fftw
                    OUTPUT_VARIABLE FFTW_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(FFTW_PREFIX)
        set(FFTW_INCLUDE_DIRS "${FFTW_PREFIX}/include")
        set(FFTW_LIBRARIES "${FFTW_PREFIX}/lib/libfftw3.dylib" "${FFTW_PREFIX}/lib/libfftw3_threads.dylib")
    endif()
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW REQUIRED fftw3)
    # Add threads library for Linux
    list(APPEND FFTW_LIBRARIES fftw3_threads)
endif()

# Find threading library (pthread on Unix systems)
find_package(Threads REQUIRED)

# Add executable
add_executable(batch_fft src/batch_fft.cpp)

# Link libraries
target_link_libraries(batch_fft
    ${FFTW_LIBRARIES}
    Threads::Threads
)

target_include_directories(batch_fft PRIVATE ${FFTW_INCLUDE_DIRS})

# Enable optimizations for release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
