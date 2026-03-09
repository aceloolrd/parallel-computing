# 05 — OpenCL

Cross-platform GPU computing using OpenCL. Same tasks as Lab 04 (CUDA), reimplemented with the OpenCL API — platform/device discovery, buffer management, kernel compilation at runtime.

## Task 1 — Image Color Quantization

Same quantization as Labs 02 and 04. OpenCL kernel compiled from source string at runtime, color palette passed via a GPU buffer.

**Input:** JPEG images at 1024×768, 1280×960, 2560×1440
**Output:** quantized images saved to `result/`

## Task 2 — Matrix Multiplication

256×256 matrices of `int16`, CPU vs OpenCL GPU. Each work-item computes one output element via `get_global_id()`.

**Approaches compared:**
- CPU scalar
- OpenCL GPU kernel

## Build

```bash
g++ -O2 -std=c++17 task1/main.cpp -lOpenCL $(pkg-config --cflags --libs opencv4) -o quantize_ocl
g++ -O2 -std=c++17 task2/main.cpp -lOpenCL -o matmul_ocl
```
