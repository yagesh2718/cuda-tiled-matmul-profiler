# CUDA Tiled Matrix Multiplication & Profiler

A CUDA C++ project that benchmarks and profiles single-precision Matrix Multiplication (SGEMM) across CPU, Naive GPU, and an Optimized Tiled GPU implementation using shared memory.

## Implementations

1. **CPU Baseline**: Sequential matrix multiplication ($i\text{-}k\text{-}j$ loop order).
2. **Naive GPU Kernel**: Each thread computes one element by reading directly from global memory.
3. **Tiled GPU Kernel**: Threads collaboratively load square tiles ($16 \times 16$) into `__shared__` memory, reducing global memory traffic and boosting throughput.

## Requirements

- NVIDIA GPU (Compute Capability $\ge 7.0$)
- CUDA Toolkit 11.4+ (with `nvcc`)
- C++17 compiler (GCC, Clang, or MSVC)
- Python 3 with `matplotlib` (optional, for plotting)

## Build & Run

### 1. Compile

```bash
# Build for default architecture (sm_80)
make

# Or specify your GPU architecture (e.g. sm_75, sm_86, sm_89)
make ARCH=sm_75
```

### 2. Run Benchmark

```bash
# Run default 1024x1024 benchmark
./matmul_profiler -s 1024

# Custom dimensions and options
./matmul_profiler -m 1024 -k 1024 -n 1024 --bench 10 --skip-cpu
```

### 3. Automated Sweep & Plotting

Run the Python harness across matrix sizes (256, 512, 1024, 2048):

```bash
python3 benchmark.py
```

This generates `benchmark_results.png` comparing execution times, GFLOPS, and speedups.

## Profiling

```bash
# Timeline profiling with NVIDIA Nsight Systems
make profile-sys

# Kernel microarchitecture profiling with NVIDIA Nsight Compute
make profile-ncu
```
