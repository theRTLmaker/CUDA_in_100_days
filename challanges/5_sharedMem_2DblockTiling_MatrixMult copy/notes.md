# Day 5 — Shared Memory 2-D Tiled Matrix Multiplication

On day 4,

---

## 🧩 Problem / Goal
Same as day 4:
A program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix `A` of `M * N` dimensions and matrix `B` of dimensions `N * K`, compute the product matrix `C`, which will have dimensions `M * K`.
All matrices are stored in row-major format.

- The final results must be stored in vector `C`

but with higher utilization of the shared memory resources for both matrices.

---

## 🧠 Key CUDA Concepts
- 2-D Block Tilling

Take advantage of the entire size of the shared memory that each SM has by loading more and doing more computation per thread.

---

## ⚙️ Kernel Overview



---

## Performance Analysis

### Running time
on H100: `` ms 🎇