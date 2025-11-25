# Day 4 — Shared Memory Tiled Matrix Multiplication

On day 3, I've greatly improved the performance of the matrix multiplication by using shared memory. However, the size of the shared memory I'm allocating is quite small compare to whats available, so the occupancy is super low. I can try to do better

---

## 🧩 Problem / Goal
Same as day 3:
A program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix `A` of `M * N` dimensions and matrix `B` of dimensions `N * K`, compute the product matrix `C`, which will have dimensions `M * K`.
All matrices are stored in row-major format.

- The final results must be stored in vector `C`

but with higher utilization of the shared memory resources for both matrices.

---

## 🧠 Key CUDA Concepts
- Block Tilling

Take advantage of the entire size of the shared memory that each SM has by loading more and doing more computation per thread.

- #pragma unroll

Use of this keyword to instruct the compiler to decompose the for loop into all the instr. It can only be done if the nunber of iterations is known beforehand

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
on H100: `48.48` ms 🎇