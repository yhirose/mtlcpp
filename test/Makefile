METAL_CPP_DIR = ../metal-cpp_macOS14.2_iOS17.2

CXX = clang++
CXXFLAGS = -std=c++20 -O2 -I$(METAL_CPP_DIR) -I../include -framework Foundation -framework Metal -framework MetalKit

INC = ../include/metal.h ../include/common.h ../include/array.h

TEST_SRC = test_metal.cpp test_array.cpp test_readme.cpp test_perceptron.cpp test.cpp
BENCH_SRC = bench.cpp
MNIST_SRC = mnist.cpp

all : test bench mnist

test : test-build
	./test-build

test-build : $(INC) $(TEST_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(TEST_SRC)

bench : bench-build
	./bench-build

bench-build : $(INC) $(BENCH_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(BENCH_SRC)

mnist : mnist-build
	./mnist-build

mnist-build : $(INC) $(MNIST_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(MNIST_SRC)

clean:
	rm -rf *-build

