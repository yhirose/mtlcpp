CXX      = clang++
CXXFLAGS = -std=c++23 -O2 -I../include -DACCELERATE_NEW_LAPACK \
           -framework Foundation -framework Metal \
           -framework Accelerate -framework MetalPerformanceShaders

INC = ../include/types.h ../include/objc.h ../include/unified_memory.h ../include/device.h ../include/cpu.h ../include/gpu.h ../include/array.h ../include/sil.h

# External libraries (comment out if not installed)
EIGEN_FLAGS = -I/opt/homebrew/include -DBENCH_HAS_EIGEN

MLX_FLAGS   = -I/opt/homebrew/include -DBENCH_HAS_MLX \
              -L/opt/homebrew/lib -lmlx -rpath /opt/homebrew/lib

TORCH_PREFIX = /opt/homebrew/opt/pytorch
TORCH_FLAGS  = -I$(TORCH_PREFIX)/include \
               -I$(TORCH_PREFIX)/include/torch/csrc/api/include \
               -DBENCH_HAS_LIBTORCH \
               -L$(TORCH_PREFIX)/lib -ltorch -ltorch_cpu -lc10 \
               -rpath $(TORCH_PREFIX)/lib

BENCH_FLAGS = $(CXXFLAGS) $(EIGEN_FLAGS) $(MLX_FLAGS) $(TORCH_FLAGS)

TARGETS = bench_sgemm bench_elementwise bench_broadcast bench_chain bench_reduction bench_mlp bench_train

all: $(TARGETS)

bench_%: bench_%.cpp bench_common.h $(INC) Makefile
	$(CXX) -o $@ $(BENCH_FLAGS) $<

# Run all or selected benchmarks: make run [BENCH="sgemm mlp"]
run: $(if $(BENCH),$(addprefix bench_,$(BENCH)),$(TARGETS))
	@for t in $^; do echo "\n=== $$t ==="; ./$$t; done

csv: $(if $(BENCH),$(addprefix bench_,$(BENCH)),$(TARGETS))
	@for t in $^; do ./$$t --csv; done

clean:
	rm -f $(TARGETS)

.PHONY: all run csv clean
