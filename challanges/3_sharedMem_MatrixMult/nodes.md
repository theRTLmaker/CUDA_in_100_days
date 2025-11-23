# Day 3 — Tiled Matrix Multiplication

On day 2, I've identified two bottleneck, low data reuse and also low data coaslescing. Let's try to fix the second one.

---

## 🧩 Problem / Goal
Same as day 2:
A program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix `A` of `M * N` dimensions and matrix `B` of dimensions `N * K`, compute the product matrix `C`, which will have dimensions `M * K`.
All matrices are stored in row-major format.

- The final results must be stored in vector `C`

but with improved memory coalescing, which hopefully will bring us added performance

---

## 🧠 Key CUDA Concepts
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


---

## Performance Analysis


### Running time