# 03 — MPI

Distributed computing using MPI (Boost.MPI). Master process coordinates workers via `MPI_Send` / `MPI_Recv`.

## Task 1 — Distributed MD5 Brute-Force

Master generates a random password (up to 5 chars, alphanumeric), computes its MD5 hash, then distributes the keyspace evenly across worker processes. Each worker searches its slice and reports back.

**Alphabet:** `0-9`, `a-z`, `A-Z` (62 characters)
**Max length:** 5 characters → up to 916M combinations

## Task 2 — Matrix Multiplication

4096×4096 matrices of `int16`. Master distributes row chunks to workers via MPI, collects partial results, and assembles the final matrix.

**Approaches compared:**
- Single-threaded
- MPI distributed
- Comparison with Labs 01 and 02

## Build

```bash
mpicxx -O2 -std=c++17 task1/main.cpp -lboost_system -o bruteforce
mpirun -np 8 ./bruteforce

mpicxx -O2 -std=c++17 task2/main.cpp -o matmul
mpirun -np 8 ./matmul
```
