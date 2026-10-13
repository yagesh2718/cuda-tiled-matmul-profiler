#include "matrix_mul.cuh"

void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float sum = 0.0f;

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_DIM + tx;
        if (row < M && a_col < K) {
            s_A[ty][tx] = A[row * K + a_col];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        int b_row = t * TILE_DIM + ty;
        if (b_row < K && col < N) {
            s_B[ty][tx] = B[b_row * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_naive(const float* d_A, const float* d_B, float* d_C,
                         int M, int K, int N, cudaStream_t stream) {
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    matmul_naive_kernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, K, N);
    CHECK_LAST_CUDA_ERROR();
}

void launch_matmul_tiled(const float* d_A, const float* d_B, float* d_C,
                         int M, int K, int N, cudaStream_t stream) {
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    matmul_tiled_kernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, K, N);
    CHECK_LAST_CUDA_ERROR();
}
