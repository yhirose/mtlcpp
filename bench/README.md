# Benchmark Suite

Silicon Array (sil) vs Eigen vs MLX vs libtorch (PyTorch C++)

## Requirements

- macOS with Apple Silicon
- Eigen: `brew install eigen`
- MLX: `brew install mlx`
- libtorch: `brew install pytorch`

If a library is not installed, comment out the corresponding `*_FLAGS` line in the Makefile.

## Build & Run

```bash
cd bench
make run
```

CSV output:

```bash
make csv
# or
./bench_sgemm --csv
```

## Benchmarks

| Name | What it measures |
|------|-----------------|
| `bench_sgemm` | Matrix multiplication (GFLOPS) at sizes 128 - 4096 |
| `bench_elementwise` | Vector add/mul throughput at 100K - 10M elements |
| `bench_broadcast` | Bias-add pattern `(N,M) + (M)` |
| `bench_mlp` | 3-layer MLP inference latency by batch size |

## Notes

- libtorch is a full deep learning framework with autograd, optimizers, and data loaders. The comparison is inherently unfair for raw computation, but illustrative of lightweight vs heavyweight tradeoffs.
- All GPU benchmarks include synchronization in timing.
- Results report best-of-N iterations after warmup.
- Eigen uses its own BLAS implementation by default. `EIGEN_USE_BLAS` with Apple Accelerate causes type conflicts, so Eigen SGEMM results reflect Eigen's built-in BLAS, not Accelerate. This makes Eigen appear significantly slower for large matrix multiplications compared to libraries that use CBLAS directly.
