# Day 3 — Tiled Matrix Multiplication

On day 2, I've identified two bottleneck, low data reuse and also low data coaslescing of matrix B. Let's try to fix the first one!

---

## 🧩 Problem / Goal
Same as day 2:
A program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix `A` of `M * N` dimensions and matrix `B` of dimensions `N * K`, compute the product matrix `C`, which will have dimensions `M * K`.
All matrices are stored in row-major format.

- The final results must be stored in vector `C`

but with improved data reuse for matrix A, which hopefully will bring us added performance

---

## 🧠 Key CUDA Concepts
- Shared Memory

Each SM - Streaming Multiprocessor (core that executes a warp) is equipped with a small shared memory (SMEM). This means tthat a thread can communicate with the other threads in its block vir the shared memory chunk. The shared memory is faster to access than L2/Main Memory, but WAY smaller, so care must be taken when populating it.

Shared memory is defined using the keyworkd `__shared__` as following:
```c++
__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
```
and the shared memory needs to be explicitly loaded from global memory by each thread
```c++
As[threadIdx.x][threadIdx.y] = A[threadIdx.x][threadIdx.y];
```

---

## ⚙️ Kernel Overview

This kernel is using a threadBlock of `16` by `16`. `16 * 16 * 4 (size of float) = 1024` << `228`KB (size of H100 shared memory). So each threadBlock will define a thread block of size 16.

Accounting for a computation where the entire matrix doesn't fit into the shared memory/one grid, I split the computations in blocks of grid size. The for loop will loud a part of the entire A and B matrices, and then compute the partial dot product accessing only the shared memory.

I applied another trick to achieve better coalescing, which was transpose B when placing it in shared memory. In this way, all the accesses to A and B are coaslesced.



---

## Performance Analysis
However, when I run the kernel, the first performance was really bad. 438.64ms on a H100.

Looking into it, it was due to bank conflicts accessing the Bs. Basically, the problem is on the access pattern:
- at fixed i, threads vary col and read `Bs[col][i]`. With BLOCK=16 and a `[16][16]` array, adjacent threads hit addresses separated by 16 floats. In 32-bank shared memory, stride 16 maps two threads per bank → 2-way conflict across the warp.

### Running time
on H100: `54.02` ms 🎇