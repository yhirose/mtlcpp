mtlcpp
======

A header-only C++20 linear algebra library for Metal on MacOS

 * This project is still in development and is far from reaching the first alpha version :)
 * Data types supported in this library are `int` and `float` only, since Metal doesn't support `double`
 * This library uses GPU cores in Apple M1 chip with [Metal-cpp](https://developer.apple.com/metal/cpp/)

Build and run unit tests and benchmark
--------------------------------------

 * Install Xcode Command Line Tools
 * Run the following commands in Terminal

```bash
cd test
make
```

Benchmark as of 3/2/2024 on M1 MacBook Pro 14
---------------------------------------------

|               ns/op |                op/s | benchmark
|--------------------:|--------------------:|:----------
|      150,856,709.00 |                6.63 | CPU: `a + b`
|        2,262,442.07 |              442.00 | GPU: `a + b`
|        1,351,401.59 |              739.97 | Eigen: `a + b`
|      964,220,500.00 |                1.04 | CPU: `a.dot(b)`
|        1,094,602.35 |              913.57 | GPU: `a.dot(b)`
|        3,002,299.36 |              333.08 | Eigen: `a * b`

```cpp
// test/bench.cpp

// `add` benchmark
const size_t n = 10'000'000;

auto a = mtl::ones<float>({n});
auto b = mtl::ones<float>({n});
auto c = mtl::array<float>();

mtl::use_cpu();
Bench().run("CPU: a + b", [&] {
  c = a + b;
});

mtl::use_gpu();
Bench().run("GPU: a + b", [&] {
  c = a + b;
});

auto aa = Eigen::Vector<float, Eigen::Dynamic>::Ones(n);
auto bb = Eigen::Vector<float, Eigen::Dynamic>::Ones(n);
auto cc = Eigen::Vector<float, Eigen::Dynamic>(n);

Bench().run("Eigen: a + b", [&] {
  cc = aa + bb;
});

// `dot` benchmark
auto a = mtl::ones<float>({1000, 1000});
auto b = mtl::ones<float>({1000, 100});
auto c = mtl::array<float>();

mtl::use_cpu();
Bench().run("CPU: a.dot(b)", [&] {
  c = a.dot(b);
});

mtl::use_gpu();
Bench().run("GPU: a.dot(b)", [&] {
  c = a.dot(b);
});

auto aa = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Ones(1000, 1000);
auto bb = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Ones(1000, 100);
auto cc = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>();

Bench().run("Eigen: a * b", [&] {
  cc = aa * bb;
});
```

Operations
----------

### GPU and CPU

 * `+` (add)
 * `-` (sub)
 * `*` (mul)
 * `/` (div)
 * `dot` (dot product)

### CPU only

 * `==`
 * `clone`
 * `constants`
 * `empty`
 * `zeros`
 * `ones`
 * `random`
 * `transpose`
 * `sigmoid`
 * `sum`
 * `mean`
 * `min`
 * `max`
 * `count`
 * `all`
 * `softmax`
 * `argmax`
 * `array_equal`
 * `allclose`

License
-------

MIT license (© 2024 Yuji Hirose)
