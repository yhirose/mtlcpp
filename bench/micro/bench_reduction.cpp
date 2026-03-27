#include <silarray.h>

#include "../bench_common.h"

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

#ifdef BENCH_HAS_GGML
#include "../bench_ggml.h"
#endif

// Reduction: sum and mean over vectors and matrices.
// Currently sil reductions are CPU-only, so this benchmarks the baseline
// and motivates GPU reduction kernels.

void bench_reduction(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("sum (1D)");
  for (auto n : {1'000'000ul, 10'000'000ul}) {
    size_t iters = n <= 1'000'000 ? 200 : 50;

    std::vector<BenchEntry> entries;

    {
      auto a = sil::random({n});
      sil::synchronize();
      volatile float s = 0;
      bench_sil(entries, iters,
                [&] { s = a.sum(); },
                [&] { s = a.sum(); });
    }

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::VectorXf aa = Eigen::VectorXf::Random(n);
      volatile float s = 0;
      entries.push_back(
          {"eigen", measure(iters, [&] { s = aa.sum(); })});
    }
#endif

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(1);
      auto* ga = inputs.new_tensor_1d(n);
      std::vector<float> ones(n, 1.0f);
      inputs.alloc_and_set(ga, ones.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx();
        auto* result = ggml_sum(ctx_g, ga);
        ggml_compute_single(result, ctx_g);
        ggml_free(ctx_g);
      })});
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
      bench_sil(entries, iters,
                [&] { s = a.sum(0); },
                [&] { s = a.sum(0); });
    }

#ifdef BENCH_HAS_EIGEN
    {
      Eigen::MatrixXf aa = Eigen::MatrixXf::Random(m, n);
      Eigen::VectorXf ss(n);
      entries.push_back(
          {"eigen", measure(iters, [&] { ss = aa.colwise().sum(); })});
    }
#endif

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(1);
      auto* ga = inputs.new_tensor_2d(n, m);
      std::vector<float> ones(m * n, 1.0f);
      inputs.alloc_and_set(ga, ones.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx();
        auto* result = ggml_sum_rows(ctx_g, ga);
        ggml_compute_single(result, ctx_g);
        ggml_free(ctx_g);
      })});
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

void bench_min_max(std::vector<BenchGroup>& groups, bool csv) {
  for (const char* op_name : {"min", "max"}) {
    if (!csv) print_section(std::format("{} (1D)", op_name).c_str());
    bool is_min = (op_name[1] == 'i');

    for (auto n : {1'000'000ul, 10'000'000ul}) {
      size_t iters = n <= 1'000'000 ? 200 : 50;

      std::vector<BenchEntry> entries;

      {
        auto a = sil::random({n});
        sil::synchronize();
        volatile float s = 0;
        bench_sil(entries, iters,
                  [&] { s = is_min ? a.min() : a.max(); },
                  [&] { s = is_min ? a.min() : a.max(); });
      }

#ifdef BENCH_HAS_EIGEN
      {
        Eigen::VectorXf aa = Eigen::VectorXf::Random(n);
        volatile float s = 0;
        entries.push_back(
            {"eigen", measure(iters, [&] { s = is_min ? aa.minCoeff() : aa.maxCoeff(); })});
      }
#endif

#ifdef BENCH_HAS_MLX
      {
        auto ma = mx::random::normal({static_cast<int>(n)});
        mx::eval(ma);
        auto ms = mx::array(0.0f);
        entries.push_back({"mlx", measure(iters, [&] {
                              ms = is_min ? mx::min(ma) : mx::max(ma);
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
               s = is_min ? a.min() : a.max();
               torch::mps::synchronize();
             })});
      }
#endif

      auto group = BenchGroup{
          std::format("{} ({})", op_name, n), std::move(entries)};
      if (!csv) print_group(group);
      groups.push_back(std::move(group));
    }
  }
}

void bench_argmax(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("argmax (2D, axis=1)");

  for (auto m : {1024ul, 4096ul}) {
    size_t n = 256;
    size_t iters = m <= 1024 ? 500 : 100;

    std::vector<BenchEntry> entries;

    {
      auto a = sil::random({m, n});
      sil::synchronize();
      auto idx = sil::array<int>();
      bench_sil(entries, iters,
                [&] { idx = a.argmax(); },
                [&] { idx = a.argmax(); });
    }

#ifdef BENCH_HAS_MLX
    {
      auto ma = mx::random::normal({static_cast<int>(m), static_cast<int>(n)});
      mx::eval(ma);
      auto ms = mx::array(0);
      entries.push_back({"mlx", measure(iters, [&] {
                            ms = mx::argmax(ma, 1);
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
             s = a.argmax(1);
             torch::mps::synchronize();
           })});
    }
#endif

    auto group = BenchGroup{
        std::format("argmax ({}x{})", m, n), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  auto mode = parse_output_mode(argc, argv);
  bool csv = (mode != OutputMode::bar);
  std::vector<BenchGroup> groups;

#ifdef BENCH_HAS_GGML
  ggml_metal_backend();  // warm up once (logs suppressed internally)
#endif

  bench_reduction(groups, csv);
  bench_min_max(groups, csv);
  bench_argmax(groups, csv);
  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "Reduction", "Reduction operations (sum, min, max, argmax) on 1D vectors and 2D matrices");
}
