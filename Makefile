NVCC        ?= nvcc
CXX         ?= g++
TARGET      ?= matmul_profiler

SRCS        := main.cu matrix_mul.cu
HEADERS     := matrix_mul.cuh
OBJS        := $(SRCS:.cu=.o)

NVCCFLAGS   := -std=c++17 -O3 --use_fast_math -lineinfo
NVCCFLAGS   += -Xcompiler "-Wall -Wextra"

ARCH        ?= sm_80

ifeq ($(GENCODE),1)
    ARCH_FLAGS := -gencode arch=compute_70,code=sm_70 \
                  -gencode arch=compute_75,code=sm_75 \
                  -gencode arch=compute_80,code=sm_80 \
                  -gencode arch=compute_86,code=sm_86 \
                  -gencode arch=compute_89,code=sm_89 \
                  -gencode arch=compute_90,code=sm_90
else
    ARCH_FLAGS := -arch=$(ARCH)
endif

DEBUG_FLAGS := -g -G -DDEBUG -O0

.PHONY: all clean run benchmark profile-sys profile-ncu debug help

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "[NVCC LINK] $@"
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $(OBJS) -o $@

%.o: %.cu $(HEADERS)
	@echo "[NVCC COMPILE] $<"
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) -c $< -o $@

debug: NVCCFLAGS := -std=c++17 $(DEBUG_FLAGS) -Xcompiler "-Wall -Wextra"
debug: clean $(TARGET)
	@echo "[BUILD] Debug binary successfully created with -g -G symbols."

run: $(TARGET)
	@echo "[RUN] Executing $(TARGET) with default 1024x1024 matrix dimensions..."
	./$(TARGET) -s 1024

benchmark: $(TARGET)
	@echo "[BENCHMARK] Executing multi-size benchmarking harness..."
	python3 benchmark.py

profile-sys: $(TARGET)
	@echo "[PROFILE] Profiling with NVIDIA Nsight Systems (nsys)..."
	nsys profile --trace=cuda,nvtx,osrt \
	             --stats=true \
	             --force-overwrite=true \
	             --output=nsys_matmul_profile \
	             ./$(TARGET) -s 1024 --warmup 2 --bench 5 --skip-cpu
	@echo "[PROFILE] Nsys report generated: nsys_matmul_profile.nsys-rep"

profile-ncu: $(TARGET)
	@echo "[PROFILE] Profiling with NVIDIA Nsight Compute (ncu)..."
	ncu --set full \
	    --target-processes all \
	    --force-overwrite \
	    --export=ncu_matmul_profile \
	    ./$(TARGET) -s 1024 --warmup 1 --bench 1 --skip-cpu
	@echo "[PROFILE] NCU report generated: ncu_matmul_profile.ncu-rep"

clean:
	@echo "[CLEAN] Removing object files and binaries..."
	rm -f $(OBJS) $(TARGET) *.nsys-rep *.ncu-rep *.sqlite benchmark_results.png

help:
	@echo "Available Makefile targets:"
	@echo "  make                 : Build the release binary ($(TARGET))"
	@echo "  make ARCH=sm_XX      : Build for specific architecture (e.g. sm_75, sm_80, sm_86, sm_89, sm_90)"
	@echo "  make GENCODE=1       : Build fatbinary for all modern NVIDIA architectures (70..90)"
	@echo "  make debug           : Build with CUDA debugging symbols (-g -G)"
	@echo "  make run             : Run matrix multiplication benchmark (1024x1024)"
	@echo "  make benchmark       : Run automated Python benchmark across multiple matrix dimensions"
	@echo "  make profile-sys     : Profile execution timeline with NVIDIA Nsight Systems"
	@echo "  make profile-ncu     : Profile kernel microarchitecture with NVIDIA Nsight Compute"
	@echo "  make clean           : Remove generated binaries and profiling reports"
