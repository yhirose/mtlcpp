# mtlcpp

A C++20 linear algebra library for Metal on MacOS

NOTE: This project is still in development and is far from reaching the first alpha version. :)

## Requirements

 - Xcode Command Line Tools

## Test

```
cd test
make test && ./test
```

## Benchmark

The benchmark provides a comparison with the Eigen and xtensor libraries.

```
brew install eigen xtensor

cd test
make bench && ./bench
```

## MNIST example

```
cd test
make mnist && ./bench
```
