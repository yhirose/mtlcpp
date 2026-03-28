#include <sil.h>

#include "bench_common.h"

#ifdef BENCH_HAS_EIGEN
#include <eigen3/Eigen/Core>
#endif

#ifdef BENCH_HAS_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

#ifdef BENCH_HAS_LIBTORCH
#include <torch/torch.h>
#endif

// Reduction: sum and mean over vectors and matrices.
// Currently sil reductions are CPU-only, so this benchmarks the baseline
// and motivates GPU reduction kernels.

void bench_reduction(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("sum (1D)");
  for (auto n : {100'000ul, 1'000'000ul, 10'000'000ul}) {
    size_t iters = n <= 100'000 ? 1000 : n <= 1'000'000 ? 200 : 50;

    std::vector<BenchEntry> entries;

    {
      auto a = sil::random({n});
      sil::synchronize();
      volatile float s = 0;
      entries.push_back(
          {"sil", measure(iters, [&] { s = a.sum(); })});
    }

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::VectorXf aa = Eigen::VectorXf::Random(n);
      volatile float s = 0;
      entries.push_back(
          {"eigen", measure(iters, [&] { s = aa.sum(); })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto ma = mx::random::normal({static_cast<int>(n)});
      mx::eval(ma);
      auto ms = mx::array(0.0f);
      entries.push_back({"mlx", measure(iters, [&] {
                            ms = mx::sum(ma);
                            mx::eval(ms);
                          })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto a = torch::randn({static_cast<long>(n)}, dev);
      auto s = torch::Tensor();
      entries.push_back(
          {"torch", measure(iters, [&] {
             s = a.sum();
             torch::mps::synchronize();
           })});
    }
#endif

    auto group = BenchGroup{
        std::format("sum ({})", n), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }

  if (!csv) print_section("sum axis=0 (2D)");
  for (auto m : {1024ul, 4096ul}) {
    size_t n = 256;
    size_t iters = m <= 1024 ? 500 : 100;

    std::vector<BenchEntry> entries;

    {
      auto a = sil::random({m, n});
      sil::synchronize();
      auto s = sil::array<float>();
      entries.push_back(
          {"sil", measure(iters, [&] { s = a.sum(0); })});
    }

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::MatrixXf aa = Eigen::MatrixXf::Random(m, n);
      Eigen::VectorXf ss(n);
      entries.push_back(
          {"eigen", measure(iters, [&] { ss = aa.colwise().sum(); })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto ma = mx::random::normal({static_cast<int>(m), static_cast<int>(n)});
      mx::eval(ma);
      auto ms = mx::array(0.0f);
      entries.push_back({"mlx", measure(iters, [&] {
                            ms = mx::sum(ma, 0);
                            mx::eval(ms);
                          })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto a = torch::randn({static_cast<long>(m), static_cast<long>(n)}, dev);
      auto s = torch::Tensor();
      entries.push_back(
          {"torch", measure(iters, [&] {
             s = a.sum(0);
             torch::mps::synchronize();
           })});
    }
#endif

    auto group = BenchGroup{
        std::format("sum axis=0 ({}x{})", m, n), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  bool csv = has_csv_flag(argc, argv);
  std::vector<BenchGroup> groups;
  bench_reduction(groups, csv);
  if (csv) print_csv(groups);
}
