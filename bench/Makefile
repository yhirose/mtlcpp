CXX      = clang++
CXXFLAGS = -std=c++23 -O2 -I../include -DACCELERATE_NEW_LAPACK \
           -framework Foundation -framework Metal \
           -framework Accelerate -framework MetalPerformanceShaders

INC = ../include/types.h ../include/objc.h ../include/unified_memory.h ../include/device.h ../include/cpu.h ../include/gpu.h ../include/array.h ../include/silarray.h

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

GGML_PREFIX = /tmp/ggml/install
GGML_FLAGS  = -I$(GGML_PREFIX)/include -DBENCH_HAS_GGML \
              -L$(GGML_PREFIX)/lib -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas \
              -framework Foundation -framework Metal -framework MetalPerformanceShaders

BENCH_FLAGS = $(CXXFLAGS) $(EIGEN_FLAGS) $(MLX_FLAGS) $(TORCH_FLAGS) $(GGML_FLAGS)

# Micro benchmarks — single operation throughput
MICRO_SRCS = $(wildcard micro/*.cpp)
MICRO_TARGETS = $(patsubst micro/%.cpp,micro/%,$(MICRO_SRCS))

# Composite benchmarks — multi-operation workloads
COMPOSITE_SRCS = $(wildcard composite/*.cpp)
COMPOSITE_TARGETS = $(patsubst composite/%.cpp,composite/%,$(COMPOSITE_SRCS))

# MNIST benchmarks — end-to-end with real data
MNIST_SRCS = $(wildcard mnist/*.cpp)
MNIST_TARGETS = $(patsubst mnist/%.cpp,mnist/%,$(MNIST_SRCS))

TARGETS = $(MICRO_TARGETS) $(COMPOSITE_TARGETS) $(MNIST_TARGETS)

all: $(TARGETS)

micro: $(MICRO_TARGETS)

composite: $(COMPOSITE_TARGETS)

mnist: $(MNIST_TARGETS)

micro/%: micro/%.cpp bench_common.h $(INC) Makefile
	$(CXX) -o $@ $(BENCH_FLAGS) $<

composite/%: composite/%.cpp bench_common.h $(INC) Makefile
	$(CXX) -o $@ $(BENCH_FLAGS) $<

mnist/%: mnist/%.cpp mnist/mnist_data.h bench_common.h $(INC) Makefile
	$(CXX) -o $@ $(BENCH_FLAGS) $<

# Run all or selected benchmarks
run: $(if $(BENCH),$(BENCH),$(TARGETS))
	@for t in $^; do echo "\n=== $$t ==="; ./$$t; done

run-micro: $(MICRO_TARGETS)
	@for t in $^; do echo "\n=== $$t ==="; ./$$t; done

run-composite: $(COMPOSITE_TARGETS)
	@for t in $^; do echo "\n=== $$t ==="; ./$$t; done

run-mnist: $(MNIST_TARGETS)
	@for t in $^; do echo "\n=== $$t ==="; ./$$t; done

csv: $(if $(BENCH),$(BENCH),$(TARGETS))
	@for t in $^; do ./$$t --csv; done

table: $(if $(BENCH),$(BENCH),$(TARGETS))
	@for t in $^; do ./$$t --table; done

clean:
	rm -f $(TARGETS)

.PHONY: all micro composite mnist run run-micro run-composite run-mnist csv table clean
