# 01 — std::thread + SIMD

CPU parallelism using `std::thread` with thread count matched to hardware concurrency. Task 2 additionally uses SSE4 SIMD intrinsics for vectorized computation.

## Task 1 — Parallel Sum

Sum of the first 10⁹ integers. Work is split evenly across threads; each thread accumulates a partial sum, then results are merged with a mutex.

**Approaches compared:**
- Single-threaded
- Multi-threaded (`std::thread`)
- Multi-threaded + SIMD (`_mm_add_epi16`, `_mm_mullo_epi16`)

## Task 2 — Matrix Multiplication

4096×4096 matrices of `int16`. Rows are distributed across threads. SIMD variant uses transposed B matrix for cache-friendly access and `_mm_mullo_epi16` for vectorized dot products.

**Approaches compared:**
- Single-threaded scalar
- Multi-threaded scalar
- Single-threaded SIMD
- Multi-threaded SIMD

## Build

```bash
g++ -O2 -std=c++17 -pthread task1/main.cpp -o sum
g++ -O2 -std=c++17 -pthread -mavx2 task2/main.cpp -o matmul
```
