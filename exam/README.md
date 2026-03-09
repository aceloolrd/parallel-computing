# Exam — 2D Convolution (OpenMP + CUDA)

2D image convolution implemented with both CPU (OpenMP) and GPU (CUDA) — the core operation behind convolutional neural networks.

## Task 1 — OpenMP Convolution

Grayscale 2D convolution parallelized with `#pragma omp parallel for collapse(2)`. Both a fixed edge-detection kernel and a random kernel are supported.

**Kernel (edge detection):**
```
-1 -1 -1
-1  8 -1
-1 -1 -1
```

## Task 2 — CUDA Convolution

Same convolution on GPU. Each CUDA thread computes one output pixel. Uses boundary clamping for border pixels.

**Config:** 32×32 thread blocks, grid sized to cover full image.

## Build

```bash
# OpenMP
g++ -O2 -std=c++17 -fopenmp task1/main.cpp $(pkg-config --cflags --libs opencv4) -o conv_omp

# CUDA
nvcc -O2 task2/main.cu $(pkg-config --cflags --libs opencv4) -o conv_cuda
```
