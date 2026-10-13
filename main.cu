#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <string>
#include <cstring>
#include "matrix_mul.cuh"

struct BenchmarkResult {
    std::string name;
    float time_ms;
    double gflops;
    double bandwidth_gb_s;
    float max_abs_diff;
    float max_rel_diff;
    bool passed;
};

namespace Color {
    const char* RESET   = "\033[0m";
    const char* BOLD    = "\033[1m";
    const char* RED     = "\033[31m";
    const char* GREEN   = "\033[32m";
    const char* YELLOW  = "\033[33m";
    const char* BLUE    = "\033[34m";
    const char* CYAN    = "\033[36m";
}

void initialize_matrix(float* mat, int rows, int cols, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dist(gen);
    }
}

bool verify_results(const float* ref, const float* test, int size,
                    float eps_abs, float eps_rel,
                    float& out_max_abs, float& out_max_rel) {
    out_max_abs = 0.0f;
    out_max_rel = 0.0f;
    bool passed = true;

    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(test[i] - ref[i]);
        float rel_diff = diff / (std::fabs(ref[i]) + 1e-7f);

        if (diff > out_max_abs) out_max_abs = diff;
        if (rel_diff > out_max_rel) out_max_rel = rel_diff;

        float max_allowed = eps_abs + eps_rel * std::fabs(ref[i]);
        if (diff > max_allowed) {
            passed = false;
        }
    }
    return passed;
}

inline double compute_gflops(int M, int K, int N, float time_ms) {
    double total_flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double seconds = static_cast<double>(time_ms) / 1000.0;
    return (total_flops / seconds) * 1e-9;
}

inline double compute_bandwidth_gb_s(int M, int K, int N, float time_ms) {
    double bytes = (static_cast<double>(M) * K +
                    static_cast<double>(K) * N +
                    static_cast<double>(M) * N) * sizeof(float);
    double seconds = static_cast<double>(time_ms) / 1000.0;
    return (bytes / seconds) * 1e-9;
}

void print_banner() {
    std::cout << Color::CYAN << Color::BOLD;
    std::cout << "================================================================================\n";
    std::cout << "     CUDA-ACCELERATED TILED MATRIX MULTIPLICATION & PROFILING ENGINE            \n";
    std::cout << "================================================================================\n";
    std::cout << Color::RESET;
}

void print_device_info() {
    int deviceId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDevice(&deviceId));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

    std::cout << Color::BOLD << "[NVIDIA GPU Hardware Context]\n" << Color::RESET;
    std::cout << "  Device Name             : " << prop.name << "\n";
    std::cout << "  Compute Capability      : " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Multiprocessors (SMs)   : " << prop.multiProcessorCount << "\n";
    std::cout << "  Total Global Memory     : " << std::fixed << std::setprecision(2)
              << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  Shared Memory per Block : " << (prop.sharedMemPerBlock / 1024.0) << " KB\n";
    std::cout << "  Warp Size               : " << prop.warpSize << " threads\n";
    std::cout << "  Tile Configuration      : " << TILE_DIM << " x " << TILE_DIM 
              << " (" << (TILE_DIM * TILE_DIM) << " threads/block)\n\n";
}

void print_usage(const char* progName) {
    std::cout << "Usage: " << progName << " [OPTIONS]\n"
              << "Options:\n"
              << "  -s, --size <N>       Square matrix dimension (default: 1024, i.e. 1024x1024)\n"
              << "  -m <M>               Matrix A rows (M)\n"
              << "  -k <K>               Shared reduction dimension (K)\n"
              << "  -n <N>               Matrix B columns (N)\n"
              << "  -w, --warmup <W>     Warmup iterations (default: 3)\n"
              << "  -b, --bench <B>      Benchmark timed iterations (default: 10)\n"
              << "  --skip-cpu           Skip CPU benchmark (useful for N >= 2048)\n"
              << "  --json               Output machine-readable JSON summary\n"
              << "  -h, --help           Display this help message\n";
}

int main(int argc, char** argv) {
    int M = 1024;
    int K = 1024;
    int N = 1024;
    int warmup_iters = 3;
    int bench_iters = 10;
    bool skip_cpu = false;
    bool output_json = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-s") == 0 || std::strcmp(argv[i], "--size") == 0) {
            if (i + 1 < argc) {
                int sz = std::atoi(argv[++i]);
                M = K = N = sz;
            }
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            K = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-w") == 0 || std::strcmp(argv[i], "--warmup") == 0) {
            if (i + 1 < argc) warmup_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-b") == 0 || std::strcmp(argv[i], "--bench") == 0) {
            if (i + 1 < argc) bench_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--skip-cpu") == 0) {
            skip_cpu = true;
        } else if (std::strcmp(argv[i], "--json") == 0) {
            output_json = true;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!output_json) {
        print_banner();
        print_device_info();
        std::cout << Color::BOLD << "[Workload Specifications]\n" << Color::RESET
                  << "  Matrix A Dimension     : " << M << " x " << K << "\n"
                  << "  Matrix B Dimension     : " << K << " x " << N << "\n"
                  << "  Matrix C Dimension     : " << M << " x " << N << "\n"
                  << "  Total Operations       : " << std::fixed << std::setprecision(2)
                  << (2.0 * M * N * K * 1e-9) << " GFLOP\n"
                  << "  Memory Footprint (A+B+C): "
                  << ((M * K + K * N + M * N) * sizeof(float) / (1024.0 * 1024.0)) << " MB\n"
                  << "  Benchmark Iterations   : " << bench_iters << " (warmup: " << warmup_iters << ")\n\n";
    }

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    std::vector<float> h_C_naive(M * N, 0.0f);
    std::vector<float> h_C_tiled(M * N, 0.0f);

    initialize_matrix(h_A.data(), M, K, 101);
    initialize_matrix(h_B.data(), K, N, 202);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, bytes_A));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, bytes_B));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, bytes_C));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    cudaEvent_t start_event, stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));

    BenchmarkResult cpu_res{"CPU Reference", 0.0f, 0.0, 0.0, 0.0f, 0.0f, true};
    BenchmarkResult naive_res{"Naive GPU Kernel", 0.0f, 0.0, 0.0, 0.0f, 0.0f, true};
    BenchmarkResult tiled_res{"Tiled GPU Kernel", 0.0f, 0.0, 0.0, 0.0f, 0.0f, true};

    if (!skip_cpu) {
        if (!output_json) {
            std::cout << Color::YELLOW << ">>> Running CPU Baseline (i-k-j loop order)... " 
                      << std::flush << Color::RESET;
        }

        auto cpu_start = std::chrono::high_resolution_clock::now();
        matmul_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, K, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
        cpu_res.time_ms = cpu_duration.count();
        cpu_res.gflops = compute_gflops(M, K, N, cpu_res.time_ms);
        cpu_res.bandwidth_gb_s = compute_bandwidth_gb_s(M, K, N, cpu_res.time_ms);

        if (!output_json) {
            std::cout << Color::GREEN << "Done (" << std::fixed << std::setprecision(2)
                      << cpu_res.time_ms << " ms, " << cpu_res.gflops << " GFLOPS)\n" << Color::RESET;
        }
    } else {
        if (!output_json) {
            std::cout << Color::YELLOW << "[!] Skipping CPU baseline execution (--skip-cpu specified)\n"
                      << Color::RESET;
        }
    }

    if (!output_json) {
        std::cout << Color::YELLOW << ">>> Benchmarking Naive GPU Kernel... " 
                  << std::flush << Color::RESET;
    }

    for (int w = 0; w < warmup_iters; ++w) {
        launch_matmul_naive(d_A, d_B, d_C, M, K, N);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float naive_total_ms = 0.0f;
    for (int b = 0; b < bench_iters; ++b) {
        CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0));
        launch_matmul_naive(d_A, d_B, d_C, M, K, N);
        CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));

        float iter_ms = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&iter_ms, start_event, stop_event));
        naive_total_ms += iter_ms;
    }

    naive_res.time_ms = naive_total_ms / bench_iters;
    naive_res.gflops = compute_gflops(M, K, N, naive_res.time_ms);
    naive_res.bandwidth_gb_s = compute_bandwidth_gb_s(M, K, N, naive_res.time_ms);

    CHECK_CUDA_ERROR(cudaMemcpy(h_C_naive.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    if (!output_json) {
        std::cout << Color::GREEN << "Done (" << std::fixed << std::setprecision(2)
                  << naive_res.time_ms << " ms, " << naive_res.gflops << " GFLOPS)\n" << Color::RESET;
    }

    if (!output_json) {
        std::cout << Color::YELLOW << ">>> Benchmarking Optimized Tiled GPU Kernel... " 
                  << std::flush << Color::RESET;
    }

    for (int w = 0; w < warmup_iters; ++w) {
        launch_matmul_tiled(d_A, d_B, d_C, M, K, N);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float tiled_total_ms = 0.0f;
    for (int b = 0; b < bench_iters; ++b) {
        CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0));
        launch_matmul_tiled(d_A, d_B, d_C, M, K, N);
        CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));

        float iter_ms = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&iter_ms, start_event, stop_event));
        tiled_total_ms += iter_ms;
    }

    tiled_res.time_ms = tiled_total_ms / bench_iters;
    tiled_res.gflops = compute_gflops(M, K, N, tiled_res.time_ms);
    tiled_res.bandwidth_gb_s = compute_bandwidth_gb_s(M, K, N, tiled_res.time_ms);

    CHECK_CUDA_ERROR(cudaMemcpy(h_C_tiled.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    if (!output_json) {
        std::cout << Color::GREEN << "Done (" << std::fixed << std::setprecision(2)
                  << tiled_res.time_ms << " ms, " << tiled_res.gflops << " GFLOPS)\n\n" << Color::RESET;
    }

    const float eps_abs = 1e-3f;
    const float eps_rel = 1e-4f;

    if (!skip_cpu) {
        naive_res.passed = verify_results(h_C_cpu.data(), h_C_naive.data(), M * N,
                                          eps_abs, eps_rel,
                                          naive_res.max_abs_diff, naive_res.max_rel_diff);

        tiled_res.passed = verify_results(h_C_cpu.data(), h_C_tiled.data(), M * N,
                                          eps_abs, eps_rel,
                                          tiled_res.max_abs_diff, tiled_res.max_rel_diff);
    } else {
        tiled_res.passed = verify_results(h_C_naive.data(), h_C_tiled.data(), M * N,
                                          eps_abs, eps_rel,
                                          tiled_res.max_abs_diff, tiled_res.max_rel_diff);
        naive_res.passed = true;
    }

    if (output_json) {
        std::cout << "{\n"
                  << "  \"dimensions\": {\"M\": " << M << ", \"K\": " << K << ", \"N\": " << N << "},\n"
                  << "  \"tile_dim\": " << TILE_DIM << ",\n"
                  << "  \"iterations\": " << bench_iters << ",\n"
                  << "  \"cpu\": {\"time_ms\": " << cpu_res.time_ms << ", \"gflops\": " << cpu_res.gflops << "},\n"
                  << "  \"naive\": {\"time_ms\": " << naive_res.time_ms << ", \"gflops\": " << naive_res.gflops 
                  << ", \"bandwidth_gb_s\": " << naive_res.bandwidth_gb_s
                  << ", \"passed\": " << (naive_res.passed ? "true" : "false")
                  << ", \"max_abs_diff\": " << naive_res.max_abs_diff << "},\n"
                  << "  \"tiled\": {\"time_ms\": " << tiled_res.time_ms << ", \"gflops\": " << tiled_res.gflops 
                  << ", \"bandwidth_gb_s\": " << tiled_res.bandwidth_gb_s
                  << ", \"passed\": " << (tiled_res.passed ? "true" : "false")
                  << ", \"max_abs_diff\": " << tiled_res.max_abs_diff << "},\n"
                  << "  \"speedup_tiled_vs_naive\": " << (naive_res.time_ms / tiled_res.time_ms) << ",\n"
                  << "  \"speedup_tiled_vs_cpu\": " << (skip_cpu ? 0.0f : (cpu_res.time_ms / tiled_res.time_ms)) << "\n"
                  << "}\n";
    } else {
        std::cout << Color::BOLD << "[Performance Summary & Accuracy Verification]\n" << Color::RESET;
        std::cout << std::string(96, '-') << "\n";
        std::cout << std::left << std::setw(20) << "Implementation"
                  << std::right << std::setw(14) << "Time (ms)"
                  << std::setw(16) << "GFLOPS"
                  << std::setw(18) << "DRAM Min (GB/s)"
                  << std::setw(14) << "Max Abs Diff"
                  << std::setw(14) << "Status" << "\n";
        std::cout << std::string(96, '-') << "\n";

        if (!skip_cpu) {
            std::cout << std::left << std::setw(20) << cpu_res.name
                      << std::right << std::setw(14) << std::fixed << std::setprecision(3) << cpu_res.time_ms
                      << std::setw(16) << std::fixed << std::setprecision(2) << cpu_res.gflops
                      << std::setw(18) << std::fixed << std::setprecision(2) << cpu_res.bandwidth_gb_s
                      << std::setw(14) << "0.000e+00"
                      << std::setw(14) << (Color::GREEN + std::string("REFERENCE") + Color::RESET) << "\n";
        }

        std::cout << std::left << std::setw(20) << naive_res.name
                  << std::right << std::setw(14) << std::fixed << std::setprecision(3) << naive_res.time_ms
                  << std::setw(16) << std::fixed << std::setprecision(2) << naive_res.gflops
                  << std::setw(18) << std::fixed << std::setprecision(2) << naive_res.bandwidth_gb_s
                  << std::setw(14) << std::scientific << std::setprecision(2) << naive_res.max_abs_diff
                  << std::setw(14) << (naive_res.passed ? (Color::GREEN + std::string("PASSED") + Color::RESET) 
                                                        : (Color::RED + std::string("FAILED") + Color::RESET)) << "\n";

        std::cout << std::left << std::setw(20) << tiled_res.name
                  << std::right << std::setw(14) << std::fixed << std::setprecision(3) << tiled_res.time_ms
                  << std::setw(16) << std::fixed << std::setprecision(2) << tiled_res.gflops
                  << std::setw(18) << std::fixed << std::setprecision(2) << tiled_res.bandwidth_gb_s
                  << std::setw(14) << std::scientific << std::setprecision(2) << tiled_res.max_abs_diff
                  << std::setw(14) << (tiled_res.passed ? (Color::GREEN + std::string("PASSED") + Color::RESET) 
                                                        : (Color::RED + std::string("FAILED") + Color::RESET)) << "\n";
        std::cout << std::string(96, '-') << "\n\n";

        std::cout << Color::BOLD << "[Speedup Multipliers]\n" << Color::RESET;
        float speedup_naive_vs_cpu = skip_cpu ? 0.0f : (cpu_res.time_ms / naive_res.time_ms);
        float speedup_tiled_vs_naive = naive_res.time_ms / tiled_res.time_ms;
        float speedup_tiled_vs_cpu = skip_cpu ? 0.0f : (cpu_res.time_ms / tiled_res.time_ms);

        if (!skip_cpu) {
            std::cout << "  Naive GPU vs CPU Baseline       : " << Color::CYAN << Color::BOLD 
                      << std::fixed << std::setprecision(2) << speedup_naive_vs_cpu << "x speedup" 
                      << Color::RESET << "\n";
        }
        std::cout << "  Tiled GPU vs Naive GPU Kernel   : " << Color::GREEN << Color::BOLD 
                  << std::fixed << std::setprecision(2) << speedup_tiled_vs_naive << "x speedup" 
                  << Color::RESET << "\n";
        if (!skip_cpu) {
            std::cout << "  Tiled GPU vs CPU Baseline       : " << Color::GREEN << Color::BOLD 
                      << std::fixed << std::setprecision(2) << speedup_tiled_vs_cpu << "x speedup" 
                      << Color::RESET << "\n";
        }
        std::cout << "\n";
    }

    CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    bool all_passed = naive_res.passed && tiled_res.passed;
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
