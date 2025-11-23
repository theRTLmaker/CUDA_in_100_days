# Day 2 — Simple Matrix Multiplication

We are stepping up to the ground base of GPGPU/ML of GPUs, multiplying matrixes. I've implemented the most basic shape of matrix multiplication, no optimization, just raw decomposition of a C code into GPU one.

---

## 🧩 Problem / Goal
A program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix `A` of `M * N` dimensions and matrix `B` of dimensions `N * K`, compute the product matrix `C`, which will have dimensions `M * K`.
All matrices are stored in row-major format.

- The final results must be stored in vector `C`

---

## 🧠 Key CUDA Concepts
- 2D Thread/block indexing: mapping thread coordinates to 2D matrix elements
- Global Memory Accesses: reading A and B, and writting C
- Grid and block geometry: using CUDA dim3 to launch a 2D grid of 16x16 threads per block
- Warping

Threads of blocks are grouped into warps, a warp is 32 threads. A warp is assigned to a warp schedular, which is a physical core that executes the instructions. The grouping into warps is based on threadId. On a multi-dimetional blockDim, threadId is computed as:
```c++
threadId = threadIdx.x +
           blockDim.x * (threadIdx.y +
                         blockDim.y * threadIdx.z)
```
Threads with neighboring threadIds are scheduled into the same warp. The important bit is that sequential memory accesses by threads belonging to the same warp can be grouped and executed as one -> COALESCING!
- Global Memory Coalescing

GPU supports 32B, 64B and 128B memort accesses of aligned accesses. Note that the acceses don't need to be sequential, just consecutive. The idea is to try to make threads that belong to the same warp share consecutive memory addresses

---

## ⚙️ Kernel Overview
This is a naive matrix multiplication kernel — each thread independently computes one output element by looping over all corresponding input elements.
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
If belongs within range of output matrix (`x < M` && `y < K`), thread idx performs
```cpp
C[idx] = SUM_o__N(A[idx] * B[idx]);
```

Noding to the global warp builder, threads with close threadIdx.x and same threadIdx.y will be grouped into the same warp. Hence, the code was written that the same warp should reuse the accesses to matrix A. To do this, the indexing of A was done using the X dimention of the threads on a block.

---

## Performance Analysis
This kernel is not efficient, but is functionally correct. As high level analysis:
- Redudant global memory loads of matrix elements (same element is load multiple times)
- Poor memory coalescing when accessing matrix B (access are being made column wise)

### Running time
on H100: `80.30` ms