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

On this kernel I made use of a bigger shared memory and increased the number of operations a threads does per load. To do that, a tiling of 8 was applied on the X axis. Meaning that each thread is now responsible for loading 8 cols of A and B.

By doing that, when doing the computation, it's possible to reuse the value of A per multiple B.

Again, to avoid bank conflicts, I padded Bs.

I used pragmas to unroll the code whenever the loop size is known at compile time.

---

## Performance Analysis
We got a slight benefit versus the older one, but this is the road.

### Running time
on H100: `48.48` ms 🎇