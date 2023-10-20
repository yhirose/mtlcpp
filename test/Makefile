METAL_CPP_DIR = ../metal-cpp_macOS13.3_iOS16.4

CXX = clang++
CXXFLAGS = -std=c++20 -O2 -I$(METAL_CPP_DIR) -I../include -framework Foundation -framework Metal -framework MetalKit

INC = ../include/metal.h ../include/compute.h ../include/array.h

TEST_INC = utils.h
TEST_SRC = test_metal.cpp test_array.cpp test_perceptron.cpp test.cpp

BENCH_SRC = bench.cpp

all : test bench
	./test
	./bench

test : $(INC) $(TEST_INC) $(TEST_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(TEST_SRC)

bench : $(INC) $(BENCH_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(BENCH_SRC)

clean:
	rm -rf test bench

