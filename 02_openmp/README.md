# 02 — OpenMP

Shared-memory parallelism using OpenMP pragmas. Thread count matched to hardware concurrency via `omp_get_max_threads()`.

## Task 1 — Image Color Quantization

Reduces an RGB image to K colors (4–10) by mapping each pixel's intensity to the nearest quantized color. Parallelized with `#pragma omp parallel for` over image rows.

**Input:** JPEG images at 1024×768, 1280×960, 2560×1440
**Output:** quantized images saved to `result/`

## Task 2 — Matrix Multiplication

4096×4096 matrices of `int16`. Outer loop parallelized with `#pragma omp parallel for`.

**Approaches compared:**
- Single-threaded
- OpenMP multi-threaded
- Comparison with Lab 01 results

## Build

```bash
g++ -O2 -std=c++17 -fopenmp task1/main.cpp $(pkg-config --cflags --libs opencv4) -o quantize
g++ -O2 -std=c++17 -fopenmp task2/main.cpp -o matmul
```
