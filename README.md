# Parallel Computing in C++

Implementations of parallel algorithms using different parallelization technologies — from CPU threading to GPU kernels.

Each section contains two tasks: a real-world problem and 4096×4096 matrix multiplication, benchmarked against a single-threaded baseline.

---

## Structure

| # | Technology | Task 1 | Task 2 |
|---|---|---|---|
| [01](01_threads/) | `std::thread` + SIMD | Parallel sum of 10⁹ integers | Matrix multiplication (int16, 4096×4096) |
| [02](02_openmp/) | OpenMP | Image color quantization | Matrix multiplication (int16, 4096×4096) |
| [03](03_mpi/) | MPI | Distributed MD5 brute-force | Matrix multiplication (int16, 4096×4096) |
| [04](04_cuda/) | CUDA | GPU image quantization | Matrix multiplication on GPU |
| [05](05_opencl/) | OpenCL | Image quantization | Matrix multiplication |
| [exam](exam/) | OpenMP + CUDA | 2D convolution (CPU) | 2D convolution (GPU) |

---

## Technologies

<img src="https://skillicons.dev/icons?i=cpp,cuda&theme=dark" />

`std::thread` · `OpenMP` · `MPI (Boost)` · `CUDA` · `OpenCL` · `SIMD (SSE4)`  · `OpenCV`

---

## Key Results

**Series sum** (10⁹ integers, 16 threads):
- Single-threaded: ~2400 ms
- Multi-threaded: ~160 ms — **15× speedup**
- Multi-threaded + SIMD: ~95 ms — **25× speedup**

**Matrix multiplication** (4096×4096, int16):
- Single-threaded: ~45 000 ms
- OpenMP (16 threads): ~3 000 ms — **15× speedup**
- CUDA: ~180 ms — **250× speedup**

**Image quantization** (2560×1440, K=10):
- CPU single-threaded: ~85 ms
- OpenMP (16 threads): ~8 ms — **10× speedup**
- CUDA kernel: ~1.2 ms — **70× speedup**
