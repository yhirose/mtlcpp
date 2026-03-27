# Benchmark Suite

Silicon Array (sil) vs Eigen vs MLX vs libtorch (PyTorch C++) vs ggml

## Requirements

- macOS with Apple Silicon
- Eigen: `brew install eigen`
- MLX: `brew install mlx`
- libtorch: `brew install pytorch`
- ggml: build from source (see below)

If a library is not installed, comment out the corresponding `*_FLAGS` line in the Makefile.

### Building ggml

```bash
git clone --depth 1 https://github.com/ggml-org/ggml.git /tmp/ggml
cd /tmp/ggml
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
cmake --install build --prefix /tmp/ggml/install
```

Then update `GGML_PREFIX` in the Makefile if installed elsewhere.

## Build & Run

```bash
cd bench
make run            # all benchmarks
make run-micro      # micro only
make run-composite  # composite only
make run-mnist      # MNIST only
```

Other output formats:

```bash
make table          # compact Markdown table (for README)
make csv            # CSV

# per-benchmark:
./micro/bench_sgemm --table
./micro/bench_sgemm --csv
```

## Benchmarks

### Micro — single operation throughput

| Name | What it measures |
|------|-----------------|
| `bench_sgemm` | Matrix multiplication (GFLOPS) at sizes 512 - 8192, square and non-square |
| `bench_elementwise` | Vector add/mul/div/pow throughput at 1M - 10M elements |
| `bench_broadcast` | Bias-add pattern `(N,M) + (M)` |
| `bench_reduction` | sum, min, max (1D), sum axis=0 (2D), argmax (2D) |
| `bench_nn_ops` | softmax, layer_norm, conv2d, batch matmul |

### Composite — multi-operation workloads

| Name | What it measures |
|------|-----------------|
| `bench_mlp` | 2-layer MLP (768->2048->768) inference |
| `bench_train` | 2-layer MLP (768->2048->768) training step |
| `bench_transformer` | Single transformer block (self-attention + FFN) inference |

### MNIST — end-to-end with real data

| Name | What it measures |
|------|-----------------|
| `bench_classifier` | 784->50->10 classifier, training + inference |
| `bench_autoencoder` | 784->512->256->64->256->512->784 autoencoder, training + inference |

## Results

Apple M1 Pro

### Broadcast

Bias-add broadcast pattern `(N,M) + (M)` at various matrix sizes

| benchmark                    | sil-gpu    | sil-cpu    | eigen      | ggml       | mlx        | torch      |
|------------------------------|------------|------------|------------|------------|------------|------------|
| broadcast (1024x1024)+(1024) | 268 us (3.6x) | **75 us**  | 92 us (1.2x) | 456 us (6.1x) | 238 us (3.2x) | 1.07 ms (14.3x) |
| broadcast (4096x512)+(512)   | 361 us (2.1x) | 185 us (1.1x) | **175 us** | 612 us (3.5x) | 332 us (1.9x) | 872 us (5.0x) |
| broadcast (4096x4096)+(4096) | **999 us** | 2.33 ms (2.4x) | 1.60 ms (1.6x) | 3.09 ms (3.2x) | **973 us** | 5.14 ms (5.3x) |

### Elementwise

Per-element vector operations (add/mul/div/pow) throughput from 1M to 10M elements

| benchmark      | sil-gpu    | sil-cpu    | eigen      | ggml       | mlx        | torch      |
|----------------|------------|------------|------------|------------|------------|------------|
| add (1000000)  | 245 us (2.3x) | **107 us** | 126 us (1.2x) | 1.34 ms (12.5x) | 268 us (2.5x) | 247 us (2.3x) |
| add (10000000) | **874 us** | 1.99 ms (2.3x) | 1.56 ms (1.8x) | 7.47 ms (8.6x) | **880 us** | **865 us** |
| mul (1000000)  | 227 us (2.4x) | **94 us**  | 124 us (1.3x) | 842 us (8.9x) | 265 us (2.8x) | 267 us (2.8x) |
| mul (10000000) | **894 us** | 1.96 ms (2.2x) | 1.55 ms (1.7x) | 7.84 ms (8.9x) | **884 us** | **904 us** |
| div (1000000)  | 250 us (2.0x) | 183 us (1.4x) | **127 us** | 1.33 ms (10.5x) | 279 us (2.2x) | 270 us (2.1x) |
| div (10000000) | **895 us** | 2.03 ms (2.3x) | 1.55 ms (1.7x) | 7.41 ms (8.3x) | **894 us** | **898 us** |
| pow (1000000)  | 266 us (1.1x) | 1.54 ms (6.3x) | 11.51 ms (47.2x) | -          | 325 us (1.3x) | **244 us** |
| pow (10000000) | **877 us** | 15.62 ms (17.8x) | 116.63 ms (133.0x) | -          | **879 us** | 979 us (1.1x) |

### NN Ops

Neural network primitives: softmax, layer normalization, 2D convolution, and batched matrix multiply

| benchmark                                      | sil-gpu    | sil-cpu    | ggml       | mlx        | torch      |
|------------------------------------------------|------------|------------|------------|------------|------------|
| softmax (256x512)                              | 244 us (1.3x) | -          | 286 us (1.5x) | **190 us** | 317 us (1.7x) |
| softmax (1024x1024)                            | 414 us (1.5x) | -          | 606 us (2.3x) | **267 us** | 530 us (2.0x) |
| softmax (4096x2048)                            | 841 us (1.2x) | -          | 2.57 ms (3.5x) | **727 us** | 1.07 ms (1.5x) |
| layer_norm (256x512)                           | 237 us (2.2x) | **110 us** | 266 us (2.4x) | 394 us (3.6x) | 230 us (2.1x) |
| layer_norm (1024x1024)                         | **445 us** | 806 us (1.8x) | 553 us (1.2x) | 811 us (1.8x) | 526 us (1.2x) |
| layer_norm (4096x2048)                         | **659 us** | 5.12 ms (7.8x) | 2.29 ms (3.5x) | 3.04 ms (4.6x) | 1.03 ms (1.6x) |
| conv2d ImageNet first layer (1x3x224x224, k=3) | -          | -          | -          | 331 us (1.1x) | **308 us** |
| conv2d ResNet mid layer (16x64x56x56, k=3)     | -          | -          | -          | 2.88 ms (1.6x) | **1.84 ms** |
| conv2d ResNet deep layer (16x128x28x28, k=3)   | -          | -          | -          | **1.92 ms** | **1.90 ms** |
| bmm attention (8 heads, seq=128, d=64)         | 260 us (13.2x) | **20 us** | 234 us (11.9x) | 199 us (10.1x) | 207 us (10.5x) |
| bmm attention (8 heads, seq=512, d=64)         | 427 us (1.3x) | **340 us** | 529 us (1.6x) | 487 us (1.4x) | 500 us (1.5x) |
| bmm attention (16 heads, seq=256, d=128)       | 580 us (2.3x) | **249 us** | 449 us (1.8x) | 464 us (1.9x) | 498 us (2.0x) |

### Reduction

Reduction operations (sum, min, max, argmax) on 1D vectors and 2D matrices

| benchmark             | sil-gpu    | sil-cpu    | eigen      | ggml       | mlx        | torch      |
|-----------------------|------------|------------|------------|------------|------------|------------|
| sum (1000000)         | 214 us (4.5x) | **47 us** | 118 us (2.5x) | 794 us (16.8x) | 255 us (5.4x) | 260 us (5.5x) |
| sum (10000000)        | 462 us (1.1x) | 665 us (1.5x) | 1.18 ms (2.7x) | 3.74 ms (8.5x) | **439 us** | 474 us (1.1x) |
| sum axis=0 (1024x256) | **20 us** | **21 us** | **20 us** | 202 us (10.1x) | 143 us (7.1x) | 166 us (8.3x) |
| sum axis=0 (4096x256) | **80 us** | **80 us** | 108 us (1.4x) | 256 us (3.2x) | 197 us (2.5x) | 223 us (2.8x) |
| min (1000000)         | **47 us** | **46 us** | 77 us (1.7x) | -          | 212 us (4.6x) | 211 us (4.5x) |
| min (10000000)        | 657 us (1.5x) | 657 us (1.5x) | 788 us (1.8x) | -          | **432 us** | 455 us (1.1x) |
| max (1000000)         | **46 us** | **46 us** | 77 us (1.7x) | -          | 201 us (4.3x) | 214 us (4.6x) |
| max (10000000)        | 660 us (1.5x) | 666 us (1.5x) | 788 us (1.8x) | -          | **435 us** | 461 us (1.1x) |
| argmax (1024x256)     | 654 us (3.4x) | 654 us (3.4x) | -          | -          | **191 us** | 206 us (1.1x) |
| argmax (4096x256)     | 2.61 ms (10.4x) | 2.61 ms (10.4x) | -          | -          | 275 us (1.1x) | **251 us** |

### SGEMM

Single-precision matrix multiplication (GFLOPS) for square matrices (512-8192) and real-world shapes

| benchmark                                             | sil-gpu    | sil-cpu    | eigen      | ggml       | mlx        | torch      |
|-------------------------------------------------------|------------|------------|------------|------------|------------|------------|
| sgemm 512x512 (2309.1 GFLOPS)                         | 332 us (2.9x) | **116 us** | 2.75 ms (23.6x) | 359 us (3.1x) | 290 us (2.5x) | 292 us (2.5x) |
| sgemm 1024x1024 (2851.9 GFLOPS)                       | **791 us** | 882 us (1.2x) | 21.97 ms (29.2x) | 965 us (1.3x) | **753 us** | 812 us (1.1x) |
| sgemm 2048x2048 (3594.9 GFLOPS)                       | **4.98 ms** | 11.12 ms (2.3x) | -          | 5.48 ms (1.1x) | **4.86 ms** | **4.78 ms** |
| sgemm 4096x4096 (3774.2 GFLOPS)                       | 39.60 ms (1.1x) | -          | -          | 38.38 ms (1.1x) | **36.60 ms** | **36.42 ms** |
| sgemm 8192x8192 (3723.8 GFLOPS)                       | 316.77 ms (1.1x) | -          | -          | 529.40 ms (1.8x) | **307.85 ms** | **295.26 ms** |
| 1x4096x4096 (single-vector inference, 58.4 GFLOPS)    | 1.08 ms (1.9x) | 1.79 ms (3.1x) | 1.61 ms (2.8x) | 628 us (1.1x) | 611 us (1.1x) | **574 us** |
| 32x4096x768 (small-batch embedding, 1159.3 GFLOPS)    | **174 us** | 626 us (3.6x) | 2.83 ms (16.3x) | 506 us (2.9x) | 359 us (2.1x) | 294 us (1.7x) |
| 256x4096x768 (medium-batch projection, 2475.0 GFLOPS) | **674 us** | 1.15 ms (1.8x) | 18.51 ms (28.5x) | 854 us (1.3x) | **651 us** | **671 us** |
| 1024x4096x768 (large-batch projection, 3317.1 GFLOPS) | **1.98 ms** | -          | -          | 2.66 ms (1.4x) | **1.94 ms** | **2.03 ms** |
| 2048x768x4096 (FFN down-projection, 3484.4 GFLOPS)    | **3.78 ms** | -          | -          | 3.89 ms (1.1x) | **3.70 ms** | **3.74 ms** |

### MLP Inference

2-layer MLP (768->2048->768 with sigmoid) forward pass — same network as training benchmark

| benchmark                  | sil-gpu    | sil-cpu    | eigen      | ggml       | mlx        | torch      |
|----------------------------|------------|------------|------------|------------|------------|------------|
| mlp inference (batch=32)   | 638 us (1.7x) | 421 us (1.1x) | 2.61 ms (7.0x) | 466 us (1.3x) | **372 us** | **383 us** |
| mlp inference (batch=64)   | 442 us (1.2x) | 503 us (1.3x) | 5.02 ms (13.1x) | 563 us (1.5x) | **383 us** | **391 us** |
| mlp inference (batch=128)  | 538 us (1.1x) | 891 us (1.8x) | 9.08 ms (18.8x) | 709 us (1.5x) | **483 us** | 529 us (1.1x) |
| mlp inference (batch=256)  | **782 us** | -          | -          | 1.07 ms (1.4x) | **780 us** | **780 us** |
| mlp inference (batch=1024) | **2.05 ms** | -          | -          | 3.37 ms (1.6x) | **2.09 ms** | **2.10 ms** |

### Training

Full training step (forward + MSE loss + manual backward + SGD update) for a 2-layer MLP (768->2048->768, sigmoid)

| benchmark              | sil-gpu    | sil-cpu    | eigen      | mlx        | torch      |
|------------------------|------------|------------|------------|------------|------------|
| train step (batch=32)  | 3.00 ms (3.1x) | 2.10 ms (2.1x) | 10.41 ms (10.6x) | **983 us** | 1.13 ms (1.1x) |
| train step (batch=64)  | 2.72 ms (1.9x) | 2.13 ms (1.5x) | 18.50 ms (12.9x) | **1.43 ms** | 1.71 ms (1.2x) |
| train step (batch=128) | 5.29 ms (3.9x) | 3.14 ms (2.3x) | -          | **1.34 ms** | 1.60 ms (1.2x) |

### Transformer

Single transformer block (multi-head self-attention + FFN) inference at various sequence lengths and model dimensions

| benchmark                     | sil-gpu    | sil-cpu    | ggml       | mlx        | torch      |
|-------------------------------|------------|------------|------------|------------|------------|
| transformer (seq=256, d=512)  | 1.41 ms (1.2x) | 13.49 ms (11.3x) | 1.66 ms (1.4x) | **1.19 ms** | 1.41 ms (1.2x) |
| transformer (seq=256, d=768)  | **2.06 ms** | 15.27 ms (7.7x) | 2.71 ms (1.4x) | **1.97 ms** | 2.14 ms (1.1x) |
| transformer (seq=256, d=1024) | **2.82 ms** | -          | 3.86 ms (1.4x) | 3.03 ms (1.1x) | 3.22 ms (1.1x) |
| transformer (seq=512, d=1024) | **4.99 ms** | -          | 8.14 ms (1.6x) | 6.02 ms (1.2x) | 5.85 ms (1.2x) |

### MNIST Classifier

784->50->10 (sigmoid, MSE loss, SGD). Training: 1 epoch, batch=100. Inference: 10000 images.

| benchmark                    | sil-gpu    | sil-cpu    | eigen      | mlx        |
|------------------------------|------------|------------|------------|------------|
| train 1 epoch (60000 images) | 1.29 s (6.2x) | 322 ms (1.5x) | **209 ms** | 235 ms (1.1x) |
| inference (10000 images)     | 810 us (1.1x) | 1.54 ms (2.2x) | 12.54 ms (17.7x) | **710 us** |

### MNIST Autoencoder

784->512->256->64->256->512->784 (sigmoid, MSE loss, SGD). Training: 1 epoch, batch=100. Inference: 10000 images.

| benchmark                    | sil-gpu    | sil-cpu    | mlx        | torch      |
|------------------------------|------------|------------|------------|------------|
| train 1 epoch (60000 images) | 3.48 s (5.6x) | 1.19 s (1.9x) | **626 ms** | 1.48 s (2.4x) |
| inference (10000 images)     | 13.02 ms (1.7x) | 33.15 ms (4.2x) | **7.83 ms** | 8.79 ms (1.1x) |

## Notes

- **sil-gpu** and **sil-cpu** show Silicon Array running on Metal GPU and Accelerate CPU respectively. Default is GPU (`use_mps()`); switch with `use_cpu()`.
- libtorch is a full deep learning framework with autograd, optimizers, and data loaders. The comparison is inherently unfair for raw computation, but illustrative of lightweight vs heavyweight tradeoffs.
- ggml is optimized for LLM inference (quantized matmul, token generation). Its graph-based API adds overhead for simple elementwise ops, making those benchmarks unrepresentative of its real-world strengths. Training benchmarks exclude ggml as it is primarily an inference engine.
- All GPU benchmarks include synchronization in timing.
- Results report best-of-N iterations after warmup.
- Eigen uses its own BLAS implementation by default. `EIGEN_USE_BLAS` with Apple Accelerate causes type conflicts, so Eigen SGEMM results reflect Eigen's built-in BLAS, not Accelerate.
- MNIST data files are expected at `test/` directory. Download from [MNIST database](http://yann.lecun.com/exdb/mnist/) if missing.
