#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCKX 32
#define BLOCKY 32
#define TILE 4

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int blockCol = blockIdx.x, blockRow = blockIdx.y;

    const int col = threadIdx.x, row = threadIdx.y;

    const int tcol = blockCol * blockDim.x * TILE + col * TILE;
    const int trow = blockRow * blockDim.y + row;

    // Create a shared memory to be used by the block
    __shared__ float As[BLOCKY][BLOCKX];
    __shared__ float Bs[BLOCKX][BLOCKX + 1];

    float threadResults[TILE] = {0.0};

    // Iterate over blocks of the matrices A and B
    // until C is fully computed
    #pragma unroll
    for (int block = 0; block < N; block += BLOCKX) {
        int nBaseShared = col * TILE;           // 0, TILE, 2*TILE, ...
        int nBaseGlobal = block + nBaseShared;  // Starting N index for this thread

        // 1-D Tiling on the X threads,
        // each loads TILE from global to shared memory
        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            int nShared = nBaseShared + t;      // 0.. BLOCKX-1
            int nGlobal = nBaseGlobal + t;      // block..block + BLOCKX-1
            // ---- A load: As[row][nShared] ----
            As[row][nShared] = trow < M && nGlobal < N ? A[trow * N + nGlobal] : 0.f;
            // ---- B load: Bs[kShared][row] (transpose) ----
            // Do a special trick, transpose B in shared memory, so B accesses are coalesced as well
            int kShared = col * TILE + t;       // 0..BLOCKX-1
            int kGlobal = tcol + t;             // columns that are load by 1-D Tiling of x
            Bs[kShared][row] = (block + row) < N && kGlobal < K ? B[(block + row) * K + kGlobal] : 0.f;
        }

        // Wait until all the data is stored in the shared memory before starting computations
        __syncthreads();

        if (tcol < K && trow < M) {
            // execute the dotproduct on the currently cached block
            for (int nLocal = 0; nLocal < min(BLOCKX, N - block); ++nLocal) {
                float A_temp = As[row][nLocal];
                #pragma unroll
                for (int t = 0; t < TILE; ++t) {
                    int kLocal = col * TILE + t;
                    int kGlobal = tcol + t;
                    if (kGlobal < K) {
                       threadResults[t] += A_temp * Bs[kLocal][nLocal];
                    }
                }
            }
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    if (tcol < K && trow < M) {
        for (int t = 0; t < min(TILE, K - tcol); ++t) {
            C[trow * K + tcol + t] = threadResults[t];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCKX/TILE, BLOCKY);
    dim3 blocksPerGrid((K + (threadsPerBlock.x * TILE) - 1) / (threadsPerBlock.x * TILE),
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
