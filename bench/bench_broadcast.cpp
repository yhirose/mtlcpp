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

void bench_broadcast(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("broadcast");
  struct Size {
    size_t rows, cols;
  };

  for (auto [rows, cols] :
       std::initializer_list<Size>{{1024, 1024}, {4096, 512}, {4096, 4096}}) {
    size_t iters = rows * cols <= 1'000'000 ? 200 : 30;

    std::vector<BenchEntry> entries;

    {
      auto a = sil::ones<float>({rows, cols});
      auto b = sil::ones<float>({cols});
      auto c = sil::array<float>();
      entries.push_back(
          {"sil", measure(iters, [&] { c = a + b; sil::synchronize(); })});
    }

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::MatrixXf aa = Eigen::MatrixXf::Ones(rows, cols);
      Eigen::VectorXf bb = Eigen::VectorXf::Ones(cols);
      Eigen::MatrixXf cc(rows, cols);
      entries.push_back(
          {"eigen",
           measure(iters, [&] { cc = aa.rowwise() + bb.transpose(); })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      int r = static_cast<int>(rows), c = static_cast<int>(cols);
      auto ma = mx::ones({r, c});
      auto mb = mx::ones({c});
      mx::eval(ma, mb);
      auto mc = mx::array(0.0f);
      entries.push_back({"mlx", measure(iters, [&] {
                            mc = mx::add(ma, mb);
                            mx::eval(mc);
                          })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      long r = rows, c = cols;
      auto a = torch::ones({r, c}, dev);
      auto b = torch::ones({c}, dev);
      auto tc = torch::Tensor();
      entries.push_back(
          {"torch", measure(iters, [&] {
             tc = torch::add(a, b);
             torch::mps::synchronize();
           })});
    }
#endif

    auto group = BenchGroup{
        std::format("broadcast ({}x{})+({})", rows, cols, cols),
        std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  bool csv = has_csv_flag(argc, argv);
  std::vector<BenchGroup> groups;
  bench_broadcast(groups, csv);
  if (csv) print_csv(groups);
}
