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

// Chained elementwise: (a - mean) * scale + bias
// Tests whether command buffer batching helps for non-matmul chains.

void bench_chain(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("chain elementwise");
  for (auto n : {100'000ul, 1'000'000ul, 10'000'000ul}) {
    size_t iters = n <= 100'000 ? 1000 : n <= 1'000'000 ? 200 : 50;

    std::vector<BenchEntry> entries;

    {
      auto a = sil::random({n});
      auto scale = sil::array<float>(2.0f);
      auto bias = sil::array<float>(0.5f);
      auto m = sil::array<float>(a.mean());
      auto c = sil::array<float>();
      entries.push_back(
          {"sil", measure(iters, [&] {
             c = (a - m) * scale + bias;
             sil::synchronize();
           })});
    }

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::VectorXf aa = Eigen::VectorXf::Random(n);
      float mean = aa.mean();
      Eigen::VectorXf cc(n);
      entries.push_back(
          {"eigen", measure(iters, [&] {
             cc = (aa.array() - mean) * 2.0f + 0.5f;
           })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto ma = mx::random::normal({static_cast<int>(n)});
      mx::eval(ma);
      auto mm = mx::mean(ma);
      mx::eval(mm);
      auto mc = mx::array(0.0f);
      entries.push_back({"mlx", measure(iters, [&] {
                            mc = mx::add(mx::multiply(mx::subtract(ma, mm), mx::array(2.0f)), mx::array(0.5f));
                            mx::eval(mc);
                          })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto a = torch::randn({static_cast<long>(n)}, dev);
      auto m = a.mean();
      auto c = torch::Tensor();
      entries.push_back(
          {"torch", measure(iters, [&] {
             c = (a - m) * 2.0f + 0.5f;
             torch::mps::synchronize();
           })});
    }
#endif

    auto group = BenchGroup{
        std::format("chain ({})", n), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  bool csv = has_csv_flag(argc, argv);
  std::vector<BenchGroup> groups;
  bench_chain(groups, csv);
  if (csv) print_csv(groups);
}
