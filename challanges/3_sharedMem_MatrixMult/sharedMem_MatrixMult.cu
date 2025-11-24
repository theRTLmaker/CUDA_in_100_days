#include <cuda_runtime.h>

#define BLOCK 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int blockCol = blockIdx.x, blockRow = blockIdx.y;
    const int col = threadIdx.x, row = threadIdx.y;
    const int tcol = blockCol * blockDim.x + col, trow = blockRow * blockDim.y + row;

    float acc = 0.f;
    // Create a shared memory to be used by the block
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK+1][BLOCK+1];

    // Iterate over blocks of the matrices A and B
    // until C is fully computed
    for (int block = 0; block < N; block += BLOCK) {
        // Load the blocks of the matrix from global to shared memory
        As[row][col] = trow < M && block + col < N ? A[trow * N + (block + col)] : 0.f;
        // Do a special trick, transpose B in shared memory, so B accesses are coalesced as well
        Bs[col][row] = block + row < N && tcol < K ? B[(block + row) * K + tcol] : 0.f;

        // Wait until all the data is stored in the shared memory before starting computations
        __syncthreads();

        if (tcol < K && trow < M) {
            // execute the dotproduct on the currently cached block
            for (int i = 0; i < min(BLOCK, N - block); ++i) {
                acc += As[row][i] * Bs[col][i];
            }
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    if (tcol < K && trow < M) {
        C[trow * K + tcol] = acc;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK, BLOCK);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
