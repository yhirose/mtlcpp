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

void bench_sgemm(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("sgemm");
  for (auto m : {128ul, 512ul, 1024ul, 2048ul, 4096ul}) {
    size_t iters = m <= 512 ? 200 : m <= 1024 ? 30 : 10;

    std::vector<BenchEntry> entries;

    {
      auto a = sil::ones<float>({m, m});
      auto b = sil::ones<float>({m, m});
      auto c = sil::array<float>();
      entries.push_back({"sil", measure(iters, [&] { c = a.dot(b); sil::synchronize(); })});
    }

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::MatrixXf aa = Eigen::MatrixXf::Ones(m, m);
      Eigen::MatrixXf bb = Eigen::MatrixXf::Ones(m, m);
      Eigen::MatrixXf cc;
      entries.push_back({"eigen", measure(iters, [&] { cc = aa * bb; })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      int mi = static_cast<int>(m);
      auto ma = mx::ones({mi, mi});
      auto mb = mx::ones({mi, mi});
      mx::eval(ma, mb);
      auto mc = mx::array(0.0f);
      entries.push_back(
          {"mlx", measure(iters, [&] {
             mc = mx::matmul(ma, mb);
             mx::eval(mc);
           })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      auto a =
          torch::ones({static_cast<long>(m), static_cast<long>(m)}, dev);
      auto b =
          torch::ones({static_cast<long>(m), static_cast<long>(m)}, dev);
      auto c = torch::Tensor();
      entries.push_back({"torch", measure(iters, [&] {
                            c = torch::mm(a, b);
                            torch::mps::synchronize();
                          })});
    }
#endif

    auto best = std::ranges::min_element(entries, {}, &BenchEntry::seconds)
                    ->seconds;

    auto group = BenchGroup{
        std::format("sgemm {}x{} ({:.1f} GFLOPS)", m, m,
                     gflops_gemm(m, m, m, best)),
        std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  bool csv = has_csv_flag(argc, argv);
  std::vector<BenchGroup> groups;
  bench_sgemm(groups, csv);
  if (csv) print_csv(groups);
}
