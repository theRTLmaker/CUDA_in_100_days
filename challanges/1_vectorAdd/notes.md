# Day 1 — Vector Addition

As the first day of this journey, nothing like doing the most basic operation possible. Starting simple: the classic "Hello World" of CUDA — a kernel that performs element-wise addition of two float vectors.

---

## 🧩 Problem / Goal
Performs element-wise addition of two vectors containing 32-bit floating point numbers on a GPU. The program should take two input vectors of equal length and produce a single output vector containing their sum.

- The final results must be stored in vector `C`

---

## 🧠 Key CUDA Concepts
How to compute a global thread index using CUDA built-ins.

How grids and blocks form a hierarchical execution model.

How to launch kernels with a 3D grid/block even when using only 1D indexing.

---

## ⚙️ Kernel Overview
Each thread is responsible for computing one element of the output vector.

int idx = blockIdx.x * blockDim.x + threadIdx.x;

If idx < N, thread idx performs
```cpp
C[idx] = A[idx] + B[idx];
```

---

## 🧵 Thread Indexing Breakdown
A kernel launch defines:

A grid: made of multiple thread blocks (can be 1D/2D/3D)

A block: made of multiple threads (can be 1D/2D/3D)

CUDA computes the thread’s global index by combining these.

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```