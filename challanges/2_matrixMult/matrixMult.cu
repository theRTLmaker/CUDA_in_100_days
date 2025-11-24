#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // compute position in C that this thread is responsible for
    const int col = blockIdx.x * blockDim.x +
              threadIdx.x;
    const int row = blockIdx.y * blockDim.y +
              threadIdx.y;

    // `if` condition is necessary for when M or K aren't multiples of 16 (number of threads in block).
    if (col < K && row < M) {
        float acc = 0.f;
        for (size_t i = 0; i < N; ++i) {
            acc += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = acc;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
