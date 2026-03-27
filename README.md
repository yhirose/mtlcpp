Silicon Array
=============

Numerical Computing Library for Apple Silicon

* Header-only C++23 library -- just `#include <silarray.h>`
* Switchable CPU/GPU backend via `sil::use_cpu()` / `sil::use_mps()` (default: GPU)
* CPU: Accelerate framework (vDSP, CBLAS, NEON)
* GPU: Metal Shading Language (MSL) for elementwise ops and matrix multiplication (STEEL kernel), Metal Performance Shaders (MPS) as fallback
* Lazy evaluation with expression templates and affine fusion for chained elementwise operations
* Data types: `float`, `int`, `bool`

Requirements
------------

* macOS with Apple Silicon
* Xcode Command Line Tools (clang++ with C++23 support)
* Frameworks: Metal, Accelerate, MetalPerformanceShaders, Foundation

Example
-------

```cpp
#include <silarray.h>

auto a = sil::ones<float>({1000, 1000});
auto b = sil::ones<float>({1000, 1000});

auto c = a + b;       // runs on GPU (default)
auto d = a.dot(b);

sil::use_cpu();        // switch to CPU backend
auto e = a + b;        // runs on CPU
```

Operations
----------

### CPU/GPU switchable

| Category | Operations |
|----------|-----------|
| Arithmetic | `+` `-` `*` `/` `pow` (elementwise, with broadcasting) |
| In-place | `+=` `-=` `*=` `/=` |
| Linear algebra | `dot` (matrix multiplication with STEEL kernel on GPU) |
| Activations | `sigmoid` `relu` `softmax` `layer_norm` |
| Fused ops | `linear` (dot + bias), `linear_sigmoid` (dot + bias + sigmoid on GPU) |
| Reduction | `sum` `sum(axis)` |

### CPU only

| Category | Operations |
|----------|-----------|
| Comparison | `==` `!=` `>` `<` `>=` `<=` |
| Shape | `clone` `transpose` `reshape` `broadcast` |
| Creation | `empty` `zeros` `ones` `random` `constants` |
| Reduction | `mean` `mean(axis)` `min` `max` `count` `all` `argmax` |
| NN utilities | `mean_square_error` `one_hot` `sigmoid_backward` |
| Selection | `where(condition, x, y)` |
| Testing | `array_equal` `allclose` |

Build and Run
-------------

### Unit tests

```bash
cd test
make
```

Tests can be run in different device modes:

```bash
./test          # GPU mode (default)
./test --gpu    # explicit GPU
./test --cpu    # CPU mode
```

### MNIST

```bash
cd test
make mnist
./mnist
```

### Benchmarks

Benchmarks compare against Eigen, MLX, libtorch, and ggml.

```bash
cd bench
make run            # all benchmarks
make run-micro      # micro only (single operation)
make run-composite  # composite only (multi-operation)
make table          # Markdown table output
```

See [bench/README.md](bench/README.md) for detailed results and setup instructions.

Architecture
------------

```
include/
  silarray.h          Main header (includes all below)
  array.h             Core array class with expression templates
  cpu.h               CPU backend (Accelerate: vDSP, CBLAS, NEON)
  gpu.h               GPU backend (Metal/MSL, STEEL matmul kernel)
  device.h            Device selection (CPU/MPS switch)
  types.h             Type concepts (float, int, bool)
  objc.h              Objective-C bridge for Metal API
  unified_memory.h    GPU shared memory management
```

License
-------

MIT license (c) 2026 Yuji Hirose
