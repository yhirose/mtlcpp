CXX = clang++
CXXFLAGS = -std=c++23 -O2 -I../include -DACCELERATE_NEW_LAPACK -framework Foundation -framework Metal -framework Accelerate -framework MetalPerformanceShaders

INC = ../include/types.h ../include/objc.h ../include/unified_memory.h ../include/device.h ../include/cpu.h ../include/gpu.h ../include/array.h ../include/sil.h

TEST_SRC = test_array.cpp test_examples.cpp test_perceptron.cpp test_2lnn.cpp test.cpp
BENCH_SRC = bench.cpp
MNIST_SRC = mnist.cpp

all : test bench mnist
	@echo "=== Auto mode ==="
	./test
	@echo "=== GPU mode ==="
	./test --gpu
	@echo "=== CPU mode ==="
	./test --cpu
	./bench
	./mnist

test : $(INC) $(TEST_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(TEST_SRC)

bench : $(INC) $(BENCH_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) -I/opt/homebrew/include -L/opt/homebrew/lib -lmlx -rpath /opt/homebrew/lib $(BENCH_SRC)

mnist : $(INC) $(MNIST_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(MNIST_SRC)

clean:
	rm -rf test bench mnist
