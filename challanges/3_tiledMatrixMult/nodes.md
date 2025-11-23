# Day 3 — Tiled Matrix Multiplication

*(Short motivation or description of the challenge.)*

---

## 🧩 Problem / Goal
Explain clearly what the kernel is supposed to do.

---

## ⚙️ Kernel Overview
Summaries such as:
- What the kernel computes
- Input/output formats
- Assumptions (e.g., matrix sizes, block size choices)

---

## 🧠 Key CUDA Concepts
- Thread/block indexing
- Memory hierarchy
- Synchronization patterns
- Any other conceptually important aspects for this day

---

## 🧵 Thread Indexing Breakdown
Explain how threads map to data (1D / 2D / 3D):

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```