Silicon Array
=============

Adaptive Numerical Computing Library for Apple Silicon

* Header-only C++23 library — just `#include <silarray.h>`
* Automatically selects the optimal backend (CPU or GPU) at runtime based on operation type and data size
* CPU: Accelerate framework (vDSP, CBLAS)
* GPU: Metal Shading Language (MSL) for elementwise ops, Metal Performance Shaders (MPS) for matrix multiplication
* Data types: `float` and `int`

Build and run unit tests and benchmark
--------------------------------------

```bash
cd test
make
```

Example
-------

```cpp
#include <silarray.h>

auto a = sil::ones<float>({1000, 1000});
auto b = sil::ones<float>({1000, 1000});

// Automatically selects CPU or GPU based on size
auto c = a + b;
auto d = a.dot(b);
```

Operations
----------

### Adaptive (CPU/GPU auto-selection)

* `+` `-` `*` `/` `pow` (elementwise arithmetic)
* `dot` (matrix multiplication)

### CPU

* `==` `!=` `>` `<` `>=` `<=`
* `clone` `transpose` `reshape`
* `constants` `empty` `zeros` `ones` `random`
* `sigmoid` `softmax`
* `sum` `mean` `min` `max` `count` `all` `argmax`
* `mean_square_error` `one_hot` `linear`
* `array_equal` `allclose`

License
-------

MIT license (c) 2026 Yuji Hirose
