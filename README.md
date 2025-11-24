# 100 Days of CUDA Kernels 🚀 ![progress badge](badge.svg)

A personal journey to explore, implement, and deeply understand GPU programming through 100 small, focused CUDA projects.

The aim is to learn GPU programming by building, testing, profiling and documenting small kernels that grow in complexity over time.

**What you'll find here**
- Daily challenge folders (`challanges/1_vectorAdd`, `challanges/2_matrixMult`, ...) each containing `notes.md` and the kernel implementation (`*.cu`).


## 💡 The Idea

Every day, I build one CUDA kernel — from the basics (*vector addition*) all the way to advanced patterns (*shared memory tiling, warp-level primitives, cooperative groups, streams, graph execution,* etc.).

This repository documents my progress with:

- 📘 **Daily Notes** — `notes.md` inside each folder
- 🧠 **Explanations** — kernels and CUDA concepts
- 🧪 **Code Implementations** — clean, runnable examples
- 📊 **A Progress Table** — tracking each challenge

The goal is not just to write kernels — it's to understand how they interact with the architecture and how to write **correct**, **fast**, and **maintainable** GPU code.


## Project goals
- Understand CUDA execution model (threads, warps, blocks, grids).
- Learn memory hierarchy and optimization: shared memory, registers, caches, and HBM/GDDR characteristics.
- Explore advanced features: cooperative groups, streams, graphs, CUDA Graphs, Tensor Cores, WMMA, and memory-bound optimizations.
- Improve profiling & benchmarking skills (nvprof / Nsight / nvtx markers).
- Produce short, self-contained notes for each day.

## 🧭 Repository Structure
```shell
CUDA_in_100_days/
├── challanges/
│   ├── ...
│   └── N_<kernel_name>/
│       ├── notes.md
│       └── <kernel_name>.cu
├── scripts/
│   └── update_readme.py
├── notes_template.md
├── badge.svg
├── README.md
└── .gitignore
```
---

## 📅 Progress Table

<!-- PROGRESS_TABLE_START -->
| Day | Folder | Topic | Short description |
|-----|--------|-------|-------------------|
| 1 | [`1_vectorAdd`](challanges/1_vectorAdd/) | Vector Addition | Basic CUDA kernel computing element-wise addition of two float vectors. |
| 2 | [`2_matrixMult`](challanges/2_matrixMult/) | Matrix Multiplication | Naive dense matrix multiplication kernel, revisiting thread indexing in 2D, memory coalescing. |
| 3 | [`3_sharedMem_MatrixMult`](challanges/3_sharedMem_MatrixMult/) | Shared Memory Matrix Multiplication |  |
| ... | ... | ... | ... |

Progress: **Day 3 / 100 (3%)**
<!-- PROGRESS_TABLE_END -->
