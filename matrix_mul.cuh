#ifndef MATRIX_MUL_CUH
#define MATRIX_MUL_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifndef TILE_DIM
#define TILE_DIM 16
#endif

#define CHECK_CUDA_ERROR(call)                                                 \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr,                                               \
                         "[CUDA ERROR] %s at %s:%d (error code %d: %s)\n",      \
                         #call, __FILE__, __LINE__, err,                       \
                         cudaGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CHECK_LAST_CUDA_ERROR()                                                \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr,                                               \
                         "[CUDA KERNEL LAUNCH ERROR] at %s:%d: %s\n",          \
                         __FILE__, __LINE__, cudaGetErrorString(err));         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N);

__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N);

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N);

void launch_matmul_naive(const float* d_A, const float* d_B, float* d_C,
                         int M, int K, int N, cudaStream_t stream = 0);

void launch_matmul_tiled(const float* d_A, const float* d_B, float* d_C,
                         int M, int K, int N, cudaStream_t stream = 0);

#endif
