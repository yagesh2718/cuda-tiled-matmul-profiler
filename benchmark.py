#!/usr/bin/env python3

import os
import sys
import json
import subprocess
import argparse
from typing import List, Dict, Any

def find_executable() -> str:
    candidates = [
        "./matmul_profiler",
        "./matmul_profiler.exe",
        "./build/matmul_profiler",
        "./build/matmul_profiler.exe"
    ]
    for c in candidates:
        if os.path.isfile(c) and (os.access(c, os.X_OK) or sys.platform == "win32"):
            return c
    return ""

def run_benchmark_for_size(exe: str, size: int, warmup: int = 2, bench: int = 5, skip_cpu: bool = False) -> Dict[str, Any]:
    cmd = [exe, "--size", str(size), "--warmup", str(warmup), "--bench", str(bench), "--json"]
    if skip_cpu:
        cmd.append("--skip-cpu")
    
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = proc.stdout.strip()
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            data = json.loads(output[json_start:json_end])
            return data
        else:
            print(f"[!] Error: Could not locate JSON output for size {size}:\n{output}", file=sys.stderr)
            return {}
    except subprocess.CalledProcessError as e:
        print(f"[!] Execution failed for size {size}: {e.stderr}", file=sys.stderr)
        return {}

def print_summary_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 98)
    print(f"{'Matrix Size':<14} | {'CPU (ms)':<12} | {'Naive (ms)':<12} | {'Tiled (ms)':<12} | "
          f"{'Naive GFLOPS':<14} | {'Tiled GFLOPS':<14} | {'Speedup (T/N)':<14}")
    print("=" * 98)
    for r in results:
        dim = r.get("dimensions", {}).get("M", 0)
        cpu_ms = r.get("cpu", {}).get("time_ms", 0.0)
        naive_ms = r.get("naive", {}).get("time_ms", 0.0)
        tiled_ms = r.get("tiled", {}).get("time_ms", 0.0)
        naive_gflops = r.get("naive", {}).get("gflops", 0.0)
        tiled_gflops = r.get("tiled", {}).get("gflops", 0.0)
        speedup = r.get("speedup_tiled_vs_naive", 1.0)
        
        cpu_str = f"{cpu_ms:.2f}" if cpu_ms > 0 else "SKIPPED"
        print(f"{dim}x{dim:<7} | {cpu_str:<12} | {naive_ms:<12.3f} | {tiled_ms:<12.3f} | "
              f"{naive_gflops:<14.2f} | {tiled_gflops:<14.2f} | {speedup:<14.2f}x")
    print("=" * 98 + "\n")

def plot_results(results: List[Dict[str, Any]], output_path: str = "benchmark_results.png"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[!] Matplotlib is not installed. Skipping plot generation.")
        print("    Install via: pip install matplotlib")
        return

    sizes = [r["dimensions"]["M"] for r in results]
    cpu_times = [r["cpu"]["time_ms"] for r in results]
    naive_times = [r["naive"]["time_ms"] for r in results]
    tiled_times = [r["tiled"]["time_ms"] for r in results]

    naive_gflops = [r["naive"]["gflops"] for r in results]
    tiled_gflops = [r["tiled"]["gflops"] for r in results]
    speedups = [r["speedup_tiled_vs_naive"] for r in results]

    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    axs[0].plot(sizes, naive_times, marker='o', linewidth=2, color='#e74c3c', label='Naive GPU Kernel')
    axs[0].plot(sizes, tiled_times, marker='s', linewidth=2, color='#2ecc71', label='Tiled GPU Kernel')
    has_cpu = any(t > 0 for t in cpu_times)
    if has_cpu:
        valid_cpu = [(s, t) for s, t in zip(sizes, cpu_times) if t > 0]
        if valid_cpu:
            cs, ct = zip(*valid_cpu)
            axs[0].plot(cs, ct, marker='^', linewidth=2, linestyle='--', color='#3498db', label='CPU Baseline (i-k-j)')
    
    axs[0].set_xscale('log', base=2)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold')
    axs[0].set_ylabel('Execution Time (ms) [Log Scale]', fontsize=11, fontweight='bold')
    axs[0].set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
    axs[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[0].legend(loc='best')
    axs[0].grid(True, which="both", ls="--", alpha=0.5)

    axs[1].plot(sizes, naive_gflops, marker='o', linewidth=2, color='#e74c3c', label='Naive GPU Kernel')
    axs[1].plot(sizes, tiled_gflops, marker='s', linewidth=2, color='#2ecc71', label='Tiled GPU Kernel')
    axs[1].set_xscale('log', base=2)
    axs[1].set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold')
    axs[1].set_ylabel('Effective Throughput (GFLOPS)', fontsize=11, fontweight='bold')
    axs[1].set_title('Compute Throughput Scaling', fontsize=12, fontweight='bold')
    axs[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[1].legend(loc='best')
    axs[1].grid(True, which="both", ls="--", alpha=0.5)

    axs[2].bar([str(s) for s in sizes], speedups, color='#9b59b6', width=0.5, edgecolor='black', alpha=0.85)
    for i, v in enumerate(speedups):
        axs[2].text(i, v + 0.1, f"{v:.2f}x", ha='center', fontweight='bold', fontsize=10)
    axs[2].set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold')
    axs[2].set_ylabel('Speedup Multiplier (x)', fontsize=11, fontweight='bold')
    axs[2].set_title('Tiled GPU Speedup vs. Naive GPU', fontsize=12, fontweight='bold')
    axs[2].grid(axis='y', ls="--", alpha=0.5)

    fig.suptitle('CUDA SGEMM Optimization & Profiling Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"[+] Performance chart saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Automated Benchmark Harness for CUDA Matrix Multiplication.")
    parser.add_argument("--exe", type=str, default="", help="Path to compiled matmul_profiler binary")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048], help="List of matrix sizes to test")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument("--bench", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument("--skip-cpu-above", type=int, default=1024, help="Skip CPU baseline for sizes strictly greater than this threshold")
    parser.add_argument("--output", type=str, default="benchmark_results.png", help="Path to save output chart")
    args = parser.parse_args()

    exe = args.exe if args.exe else find_executable()
    if not exe:
        print("[!] Error: Could not locate matmul_profiler executable.", file=sys.stderr)
        print("    Please compile the project first using 'make' or provide --exe path.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Found executable: {exe}")
    print(f"[*] Benchmarking matrix sizes: {args.sizes}")

    all_results = []
    for sz in args.sizes:
        skip_cpu = sz > args.skip_cpu_above
        print(f"[*] Testing {sz}x{sz} (skip_cpu={skip_cpu})...")
        res = run_benchmark_for_size(exe, sz, args.warmup, args.bench, skip_cpu)
        if res:
            all_results.append(res)

    if all_results:
        print_summary_table(all_results)
        plot_results(all_results, args.output)
    else:
        print("[!] No benchmark results collected.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
