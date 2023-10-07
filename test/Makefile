METAL_CPP_DIR = ../metal-cpp_macOS13.3_iOS16.4

CXX = clang++
CXXFLAGS = -std=c++20 -O2 -I$(METAL_CPP_DIR) -I../include -framework Foundation -framework Metal -framework MetalKit

INC = ../include/metal.h ../include/gpu.h ../include/array.h
SRC = ../src/metal.cpp ../src/gpu.cpp

TEST_SRC = test_metal.cpp test_gpu.cpp test_array.cpp test.cpp

BENCH_SRC = bench.cpp

all : test bench

test : $(SRC) $(INC) $(TEST_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(SRC) $(TEST_SRC)

bench : $(SRC) $(INC) $(BENCH_SRC) Makefile
	$(CXX) -o $@ $(CXXFLAGS) $(SRC) $(BENCH_SRC)

clean:
	rm -rf test bench

