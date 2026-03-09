# 04 — CUDA

GPU computing using CUDA. Kernels run on device with 16×16 thread blocks; quantized color palette stored in constant memory for fast access.

## Task 1 — GPU Image Quantization

Same quantization task as Lab 02, but executed entirely on GPU. Each CUDA thread processes one pixel independently.

**Optimizations:**
- Constant memory for color palette (`__constant__`)
- Coalesced memory access (row-major pixel layout)
- 16×16 thread blocks

**Input:** JPEG images at 1024×768, 1280×960, 2560×1440
**Output:** quantized images saved to `result/`

## Task 2 — Matrix Multiplication on GPU

4096×4096 matrix multiplication using CUDA kernels. Each thread computes one output element.

**Approaches compared:**
- CPU single-threaded
- CUDA kernel
- Comparison with Labs 01–03

## Build

Requires CUDA Toolkit and OpenCV built with CUDA support.

```bash
nvcc -O2 task1/main.cu $(pkg-config --cflags --libs opencv4) -o quantize_cuda
nvcc -O2 task2/main.cu -o matmul_cuda
```
