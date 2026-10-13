cmake_minimum_required(VERSION 3.18)
project(cuda-tiled-matmul-profiler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

set(SOURCES
    main.cu
    matrix_mul.cu
)

set(HEADERS
    matrix_mul.cuh
)

add_executable(matmul_profiler ${SOURCES} ${HEADERS})

target_include_directories(matmul_profiler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(matmul_profiler PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
