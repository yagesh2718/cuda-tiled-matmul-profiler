# CUDA-Accelerated Tiled Matrix Multiplication & Profiling Engine

[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-17-00599C?logo=c%2B%2B&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![Nsight Systems](https://img.shields.io/badge/Nsight%20Systems-Profile-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/nsight-systems)
[![Nsight Compute](https://img.shields.io/badge/Nsight%20Compute-Microarchitecture-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/nsight-compute)

A high-performance CUDA C++ systems project implementing, benchmarking, and profiling single-precision General Matrix Multiplication (**SGEMM**) across three architectures:
1. **CPU Reference**: Cache-optimized $i\text{-}k\text{-}j$ traversal baseline.
2. **Naive GPU Kernel**: Per-thread global memory evaluation (bandwidth-bound).
3. **Optimized Tiled GPU Kernel**: Collaborative on-chip `__shared__` memory staging with conflict-free bank access and dual-barrier synchronization.

Includes an automated Python benchmarking and visualization suite alongside deep-dive profiling recipes for **NVIDIA Nsight Systems (`nsys`)** and **NVIDIA Nsight Compute (`ncu`)**.

---

## Table of Contents

- [Architectural Overview](#architectural-overview)
  - [The Memory Wall & Arithmetic Intensity](#the-memory-wall--arithmetic-intensity)
  - [Naive Kernel vs. Tiled Shared Memory Kernel](#naive-kernel-vs-tiled-shared-memory-kernel)
  - [Dual Barrier Synchronization Rationale](#dual-barrier-synchronization-rationale)
  - [Bank Conflict-Free Memory Access](#bank-conflict-free-memory-access)
- [Project Structure](#project-structure)
- [Prerequisites & Requirements](#prerequisites--requirements)
- [Build Instructions](#build-instructions)
  - [Using Make (Recommended)](#using-make-recommended)
  - [Direct NVCC Compilation](#direct-nvcc-compilation)
  - [Using CMake](#using-cmake)
- [Running the Profiler](#running-the-profiler)
  - [Command-Line Options](#command-line-options)
  - [Sample Output](#sample-output)
- [Automated Benchmarking Suite](#automated-benchmarking-suite)
- [Deep-Dive Profiling Guide](#deep-dive-profiling-guide)
  - [1. NVIDIA Nsight Systems (`nsys`)](#1-nvidia-nsight-systems-nsys)
  - [2. NVIDIA Nsight Compute (`ncu`)](#2-nvidia-nsight-compute-ncu)
- [Expected Performance & Roofline Analysis](#expected-performance--roofline-analysis)

---

## Architectural Overview

Matrix multiplication ($C = A \times B$, where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, and $C \in \mathbb{R}^{M \times N}$) is the primary compute primitive powering modern Deep Learning (MLPs, Transformers, Convolutions). Optimizing SGEMM requires bridging the gap between GPU compute capability (TeraFLOPS) and global memory bandwidth (GB/s).

```
   Matrix A [M x K]          Matrix B [K x N]               Matrix C [M x N]
+--------------------+    +--------------------+        +--------------------+
|                    |    |      Tile B        |        |                    |
|--------------------|    |      [Tk x Tn]     |        |                    |
|    Tile A [Tm x Tk]|    +--------------------+   ->   |    Tile C [Tm x Tn]|
|--------------------|    |                    |        |                    |
|                    |    |                    |        +--------------------+
+--------------------+    +--------------------+
```

### The Memory Wall & Arithmetic Intensity

Each element in $C$ requires $K$ floating-point multiplications and $K$ additions, totaling:
$$\text{Total FLOPs} = 2 \cdot M \cdot N \cdot K$$

- **Global Memory Latency**: High-Bandwidth Memory (HBM/GDDR6X) incurs **200–400 clock cycles** of access latency.
- **Shared Memory Latency**: On-chip SRAM incurs **~20–30 clock cycles** of latency, with aggregate bandwidth exceeding **15–20 TB/s** on modern architectures.

#### Naive Kernel
Every thread computes a single element $C[i, j]$ by loading row $i$ of $A$ and column $j$ of $B$ directly from DRAM:
- Total DRAM Reads: $2 \cdot M \cdot N \cdot K \cdot \text{sizeof(float)}$ bytes.
- **Arithmetic Intensity**:
  $$\text{AI}_{\text{naive}} = \frac{2 \cdot M \cdot N \cdot K \text{ FLOPs}}{2 \cdot M \cdot N \cdot K \cdot 4 \text{ bytes}} = 0.25 \text{ FLOP/byte}$$
- On an NVIDIA RTX 4090 (~1,008 GB/s DRAM bandwidth), the naive kernel is bound to a theoretical ceiling of at most:
  $$1008 \text{ GB/s} \times 0.25 \text{ FLOP/byte} \approx 252 \text{ GFLOPS}$$
  Even though the GPU hardware can execute over **80,000 GFLOPS (80 TFLOPS)** of single-precision math! The compute engines sit starved of data.

#### Tiled Shared Memory Kernel
By staging square sub-matrices of dimensions $\text{TILE\_DIM} \times \text{TILE\_DIM}$ into fast `__shared__` memory:
- Threads in a thread block collaborate to load each tile once into shared memory.
- Once staged, each loaded float is reused $\text{TILE\_DIM}$ times across the threads of the block.
- Total DRAM traffic is reduced by a factor of $\text{TILE\_DIM}$ (e.g., $16\times$ reduction for $16 \times 16$ tiles).
- **Arithmetic Intensity**:
  $$\text{AI}_{\text{tiled}} = \frac{2 \cdot \text{TILE\_DIM} \cdot \text{TILE\_DIM} \cdot \text{TILE\_DIM} \text{ FLOPs}}{2 \cdot \text{TILE\_DIM} \cdot \text{TILE\_DIM} \cdot 4 \text{ bytes}} = \frac{\text{TILE\_DIM}}{4} = 4.0 \text{ FLOP/byte (for tile size 16)}$$

---

### Dual Barrier Synchronization Rationale

In `matmul_tiled_kernel`, two distinct calls to `__syncthreads()` are strictly mandatory within each phase:

```cuda
for (int t = 0; t < numTiles; ++t) {
    // 1. Collaborative load from global DRAM -> shared memory
    s_A[ty][tx] = A[...];
    s_B[ty][tx] = B[...];

    __syncthreads(); // [BARRIER 1: Read-After-Write (RAW) Hazard Elimination]

    // 2. Perform fused multiply-accumulate on on-chip tiles
    for (int k = 0; k < TILE_DIM; ++k) {
        sum += s_A[ty][k] * s_B[k][tx];
    }

    __syncthreads(); // [BARRIER 2: Write-After-Read (WAR) Hazard Elimination]
}
```

1. **Barrier 1 (RAW Barrier)**:
   Prevents threads from computing with uninitialized or partially loaded shared memory tiles. All threads must finish writing their assigned element to `s_A` and `s_B` before any thread starts the accumulation loop.
2. **Barrier 2 (WAR Barrier)**:
   Prevents faster threads from looping around and overwriting `s_A` and `s_B` for phase $t+1$ while slower threads in the same block are still calculating phase $t$.

---

### Bank Conflict-Free Memory Access

CUDA shared memory is organized into **32 independent banks** (each 4 bytes wide). Consecutive 32-bit words map to successive banks:
$$\text{Bank ID} = (\text{Byte Address} / 4) \pmod{32}$$

In the inner accumulation loop:
- **Matrix A tile (`s_A[ty][k]`)**:
  Within a warp, all threads executing concurrently share the same row index `ty` (or identical `ty` ranges across lanes). For a given loop step `k`, multiple threads read the **exact same address** `s_A[ty][k]`. The GPU shared memory controller handles identical address requests via **hardware broadcast**, consuming zero additional cycles (no conflict).
- **Matrix B tile (`s_B[k][tx]`)**:
  Threads within a warp have consecutive column indices `tx` $\in [0, 15]$. Therefore, thread $i$ accesses bank $i \pmod{32}$. Every active thread accesses a distinct memory bank simultaneously, achieving **100% conflict-free single-cycle shared memory throughput**.

---

## Project Structure

```
cuda-tiled-matmul-profiler/
├── matrix_mul.cuh       # Header: definitions, TILE_DIM, CHECK_CUDA_ERROR macros
├── matrix_mul.cu        # Implementation: CPU matmul, Naive kernel, Tiled kernel
├── main.cu              # Harness: CUDA events, memory management, CLI & verification
├── Makefile             # Multi-target compilation (sm_70 to sm_90, debug, nsys, ncu)
├── CMakeLists.txt       # Modern CMake configuration
├── benchmark.py         # Automated Python multi-size benchmarking and plotting suite
└── README.md            # Architectural deep dive and profiling guide
```

---

## Prerequisites & Requirements

- **NVIDIA GPU**: Compute Capability $\ge 7.0$ (Volta, Turing, Ampere, Ada Lovelace, Hopper, Blackwell).
- **CUDA Toolkit**: Version 11.4 or newer (12.x recommended).
- **C++ Compiler**: GCC $\ge 9$, Clang $\ge 11$, or MSVC $\ge 2019$ supporting C++17.
- **Python 3.8+** (Optional, for `benchmark.py`):
  ```bash
  pip install matplotlib numpy
  ```
- **NVIDIA Profilers**:
  - `nsys` (NVIDIA Nsight Systems)
  - `ncu` (NVIDIA Nsight Compute)

---

## Build Instructions

### Using Make (Recommended)

Build release binary targeted for NVIDIA Ampere (`sm_80`):
```bash
make
```

Build for a specific GPU architecture:
```bash
# Turing (RTX 2080, T4)
make ARCH=sm_75

# Ampere (RTX 3080, A100)
make ARCH=sm_80

# Ada Lovelace (RTX 4090, L40)
make ARCH=sm_89

# Hopper (H100)
make ARCH=sm_90
```

Build a universal multi-architecture fatbinary:
```bash
make GENCODE=1
```

Build with debug symbols for `cuda-gdb` and `compute-sanitizer`:
```bash
make debug
```

---

### Direct NVCC Compilation

```bash
nvcc -std=c++17 -O3 --use_fast_math -arch=sm_80 main.cu matrix_mul.cu -o matmul_profiler
```

---

### Using CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80"
cmake --build . -j$(nproc)
```

---

## Running the Profiler

### Command-Line Options

```
Usage: ./matmul_profiler [OPTIONS]
Options:
  -s, --size <N>       Square matrix dimension (default: 1024, i.e. 1024x1024)
  -m <M>               Matrix A rows (M)
  -k <K>               Shared reduction dimension (K)
  -n <N>               Matrix B columns (N)
  -w, --warmup <W>     Warmup iterations (default: 3)
  -b, --bench <B>      Benchmark timed iterations (default: 10)
  --skip-cpu           Skip CPU benchmark (useful for N >= 2048)
  --json               Output machine-readable JSON summary
  -h, --help           Display this help message
```

Execute default $1024 \times 1024$ benchmark:
```bash
./matmul_profiler -s 1024
```

Execute non-square matrix multiplication ($M=2048, K=1024, N=4096$) without CPU:
```bash
./matmul_profiler -m 2048 -k 1024 -n 4096 --skip-cpu --bench 20
```

---

### Sample Output

```
================================================================================
     CUDA-ACCELERATED TILED MATRIX MULTIPLICATION & PROFILING ENGINE            
================================================================================
[NVIDIA GPU Hardware Context]
  Device Name             : NVIDIA GeForce RTX 3080
  Compute Capability      : 8.6
  Multiprocessors (SMs)   : 68
  Total Global Memory     : 10.00 GB
  Shared Memory per Block : 48.00 KB
  Warp Size               : 32 threads
  Tile Configuration      : 16 x 16 (256 threads/block)

[Workload Specifications]
  Matrix A Dimension     : 1024 x 1024
  Matrix B Dimension     : 1024 x 1024
  Matrix C Dimension     : 1024 x 1024
  Total Operations       : 2.15 GFLOP
  Memory Footprint (A+B+C): 12.00 MB
  Benchmark Iterations   : 10 (warmup: 3)

>>> Running CPU Baseline (i-k-j loop order)... Done (312.45 ms, 6.87 GFLOPS)
>>> Benchmarking Naive GPU Kernel... Done (3.82 ms, 562.14 GFLOPS)
>>> Benchmarking Optimized Tiled GPU Kernel... Done (0.91 ms, 2359.87 GFLOPS)

[Performance Summary & Accuracy Verification]
------------------------------------------------------------------------------------------------
Implementation               Time (ms)          GFLOPS     DRAM Min (GB/s)  Max Abs Diff        Status
------------------------------------------------------------------------------------------------
CPU Reference                  312.450            6.87               40.28     0.000e+00     REFERENCE
Naive GPU Kernel                 3.821          562.14             3293.90     1.240e-04        PASSED
Tiled GPU Kernel                 0.910         2359.87            13832.97     1.240e-04        PASSED
------------------------------------------------------------------------------------------------

[Speedup Multipliers]
  Naive GPU vs CPU Baseline       : 81.77x speedup
  Tiled GPU vs Naive GPU Kernel   : 4.20x speedup
  Tiled GPU vs CPU Baseline       : 343.35x speedup
```

---

## Automated Benchmarking Suite

An automated Python benchmarking harness (`benchmark.py`) sweeps matrix dimensions ($256, 512, 1024, 2048$), validates numerical correctness, aggregates execution statistics via JSON, and produces an exportable visualization chart:

```bash
python3 benchmark.py --sizes 256 512 1024 2048 --warmup 2 --bench 5
```

The script outputs `benchmark_results.png` showing:
1. **Execution Time (ms)** across matrix sizes on a log-log scale.
2. **Computational Throughput (GFLOPS)** scaling curves.
3. **Speedup Multipliers** of Tiled Shared Memory over the Naive implementation.

---

## Deep-Dive Profiling Guide

### 1. NVIDIA Nsight Systems (`nsys`)

Nsight Systems visualizes system-wide timeline traces, memory transfer bottlenecks, and host-device synchronization stalls.

#### Command
```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --force-overwrite=true \
    --output=nsys_matmul_profile \
    ./matmul_profiler -s 1024 --warmup 2 --bench 5 --skip-cpu
```
*(Or simply run: `make profile-sys`)*

#### Timeline Key Metrics to Inspect in Nsight Systems GUI:
1. **CUDA API Overhead**: Verify that `cudaMalloc` and initial page-table faults (`cudaMemcpy`) occur during setup, outside the timing loop.
2. **Kernel Execution Stream**: Check that `matmul_tiled_kernel` displays consistent, uninterrupted execution blocks without CPU synchronization gaps.
3. **Event Timing Accuracy**: Compare `cudaEventElapsedTime` with the OS runtime duration shown on the CPU timeline.

---

### 2. NVIDIA Nsight Compute (`ncu`)

Nsight Compute profiles microarchitectural behavior inside the Streaming Multiprocessor (SM), measuring cache hit rates, memory rooflines, and pipeline stalls.

#### Command
```bash
ncu --set full \
    --target-processes all \
    --force-overwrite \
    --export=ncu_matmul_profile \
    ./matmul_profiler -s 1024 --warmup 1 --bench 1 --skip-cpu
```
*(Or simply run: `make profile-ncu`)*

#### Critical Metrics to Compare:

| Profiler Metric | Naive GPU Kernel | Tiled Shared Memory Kernel |
| :--- | :--- | :--- |
| **Speed of Light (Compute %)** | Low (< 20%) | High (> 70%) |
| **Speed of Light (Memory %)** | High (> 85% DRAM Bound) | Shifted to Shared Memory / L1 |
| **DRAM Read Bandwidth** | Approaching device saturation | Reduced by $\approx 16\times$ |
| **Shared Memory Throughput** | 0 GB/s | Multi-TB/s |
| **Warp Stall Reasons** | `Stall Long Scoreboard` (DRAM latency) | `Stall Sync` / `Stall Math Pipe` |
| **Shared Bank Conflicts** | None | 0 conflicts (Broadcast & Bank-aligned) |

To capture specific isolated metrics directly in the terminal:
```bash
# Inspect DRAM throughput and roofline positioning:
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./matmul_profiler -s 1024 --skip-cpu
```

---

## Expected Performance & Roofline Analysis

- **Arithmetic Intensity Boost**: Moving from the naive kernel to $16 \times 16$ tiling increases arithmetic intensity from **0.25 FLOP/byte to 4.0 FLOP/byte** ($16\times$ improvement).
- **Latency Hiding**: Tiled shared memory eliminates long-scoreboard stalls caused by high-latency global memory fetches, allowing warp schedulers to saturate the floating-point arithmetic pipelines.
- **Scalability**: For large matrices ($N \ge 1024$), the tiled kernel consistently achieves **$3\times$ to $5\times$ speedup over the naive GPU kernel** and **$>300\times$ speedup over single-threaded CPU**.

