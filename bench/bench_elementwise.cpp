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

void bench_op(std::vector<BenchGroup>& groups, bool csv, const char* op_name,
              auto sil_fn, auto eigen_fn, auto mlx_fn, auto torch_fn) {
  for (auto n : {100'000ul, 1'000'000ul, 10'000'000ul}) {
    size_t iters = n <= 100'000 ? 1000 : n <= 1'000'000 ? 200 : 50;

    std::vector<BenchEntry> entries;

    {
      auto a = sil::ones<float>({n});
      auto b = sil::ones<float>({n});
      auto c = sil::array<float>();
      entries.push_back(
          {"sil", measure(iters, [&] { c = sil_fn(a, b); })});
    }

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::VectorXf aa = Eigen::VectorXf::Ones(n);
      Eigen::VectorXf bb = Eigen::VectorXf::Ones(n);
      Eigen::VectorXf cc(n);
      entries.push_back(
          {"eigen", measure(iters, [&] { eigen_fn(cc, aa, bb); })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto ma = mx::ones({static_cast<int>(n)});
      auto mb = mx::ones({static_cast<int>(n)});
      mx::eval(ma, mb);
      auto mc = mx::array(0.0f);
      entries.push_back({"mlx", measure(iters, [&] {
                            mc = mlx_fn(ma, mb);
                            mx::eval(mc);
                          })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto a = torch::ones({static_cast<long>(n)}, dev);
      auto b = torch::ones({static_cast<long>(n)}, dev);
      auto c = torch::Tensor();
      entries.push_back(
          {"torch", measure(iters, [&] {
             c = torch_fn(a, b);
             torch::mps::synchronize();
           })});
    }
#endif

    auto group = BenchGroup{std::format("{} ({})", op_name, n), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  bool csv = has_csv_flag(argc, argv);
  std::vector<BenchGroup> groups;

  if (!csv) print_section("add");
  bench_op(
      groups, csv, "add",
      [](auto& a, auto& b) { return a + b; },
#ifdef BENCH_HAS_EIGEN
      [](auto& c, auto& a, auto& b) { c = a + b; },
#else
      [](auto&, auto&, auto&) {},
#endif
#ifdef BENCH_HAS_MLX
      [](auto& a, auto& b) { return mx::add(a, b); },
#else
      [](auto&, auto&) { return 0; },
#endif
#ifdef BENCH_HAS_LIBTORCH
      [](auto& a, auto& b) { return torch::add(a, b); }
#else
      [](auto&, auto&) { return 0; }
#endif
  );

  if (!csv) print_section("mul");
  bench_op(
      groups, csv, "mul",
      [](auto& a, auto& b) { return a * b; },
#ifdef BENCH_HAS_EIGEN
      [](auto& c, auto& a, auto& b) { c = a.cwiseProduct(b); },
#else
      [](auto&, auto&, auto&) {},
#endif
#ifdef BENCH_HAS_MLX
      [](auto& a, auto& b) { return mx::multiply(a, b); },
#else
      [](auto&, auto&) { return 0; },
#endif
#ifdef BENCH_HAS_LIBTORCH
      [](auto& a, auto& b) { return torch::mul(a, b); }
#else
      [](auto&, auto&) { return 0; }
#endif
  );

  if (csv) print_csv(groups);
}
