# Day 2 — Simple Matrix Multiplication

We are stepping up to the ground base of GPGPU/ML of GPUs, multiplying matrixes. I've implemented the most basic shape of matrix multiplication, no optimization, just raw decomposition of a C code into GPU one.

---

## 🧩 Problem / Goal
A program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix `A` of `M * N` dimensions and matrix `B` of dimensions `N * K`, compute the product matrix `C`, which will have dimensions `M * K`.
All matrices are stored in row-major format.

- The final results must be stored in vector `C`

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

---

## 🧠 Key CUDA Concepts
- 2D Thread/block indexing: mapping thread coordinates to 2D matrix elements
- Global Memory Accesses: reading A and B, and writting C
- Grid and block geometry: using CUDA dim3 to launch a 2D grid of 16x16 threads per blocl

---

## Performance Analysis
This kernel is not efficient, but is functionally correct. As high level analysis:
- Redudant global memory loads of matrix elements (same element is load multiple times)
- Poor memory coalescing when accessing matrix B (access are being made column wise)