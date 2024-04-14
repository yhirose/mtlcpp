METAL_CPP_DIR = ../metal-cpp_macOS14.2_iOS17.2

CXX = clang++
CXXFLAGS = -std=c++20 -O2 -I$(METAL_CPP_DIR) -I../include -framework Foundation -framework Metal -framework MetalKit

INC = ../include/metal.h ../include/array.h ../include/mtlcpp.h

TEST_SRC = test_array.cpp test_examples.cpp test_perceptron.cpp test.cpp
BENCH_SRC = bench.cpp
MNIST_SRC = mnist.cpp

all : test bench mnist
	./test
	./bench
	./mnist

test : $(INC) $(TEST_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(TEST_SRC)

bench : $(INC) $(BENCH_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(BENCH_SRC)

mnist : $(INC) $(MNIST_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(MNIST_SRC)

clean:
	rm -rf test bench mnist

