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

// Network: 784 -> 256 -> sigmoid -> 128 -> sigmoid -> 10

void bench_mlp(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("mlp inference");
  for (auto batch : {1ul, 32ul, 128ul, 512ul}) {
    size_t iters = batch <= 32 ? 500 : 100;

    std::vector<BenchEntry> entries;

    {
      auto x = sil::random({batch, 784});
      auto W1 = sil::random({784, 256});
      auto b1 = sil::zeros<float>({256});
      auto W2 = sil::random({256, 128});
      auto b2 = sil::zeros<float>({128});
      auto W3 = sil::random({128, 10});
      auto b3 = sil::zeros<float>({10});

      entries.push_back({"sil", measure(iters, [&] {
                            auto h1 = x.linear(W1, b1).sigmoid();
                            auto h2 = h1.linear(W2, b2).sigmoid();
                            auto out = h2.linear(W3, b3);
                          })});
    }

#ifdef BENCH_HAS_EIGEN
    {
      auto sigmoid = [](const Eigen::MatrixXf& x) -> Eigen::MatrixXf {
        return (1.0f + (-x).array().exp()).inverse().matrix();
      };

      Eigen::MatrixXf x = Eigen::MatrixXf::Random(batch, 784);
      Eigen::MatrixXf W1 = Eigen::MatrixXf::Random(784, 256);
      Eigen::VectorXf b1 = Eigen::VectorXf::Zero(256);
      Eigen::MatrixXf W2 = Eigen::MatrixXf::Random(256, 128);
      Eigen::VectorXf b2 = Eigen::VectorXf::Zero(128);
      Eigen::MatrixXf W3 = Eigen::MatrixXf::Random(128, 10);
      Eigen::VectorXf b3 = Eigen::VectorXf::Zero(10);

      entries.push_back({"eigen", measure(iters, [&] {
                            auto h1 = sigmoid((x * W1).rowwise() + b1.transpose());
                            auto h2 = sigmoid((h1 * W2).rowwise() + b2.transpose());
                            Eigen::MatrixXf out = (h2 * W3).rowwise() + b3.transpose();
                          })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      int b = static_cast<int>(batch);
      auto x = mx::random::normal({b, 784});
      auto W1 = mx::random::normal({784, 256});
      auto b1 = mx::zeros({256});
      auto W2 = mx::random::normal({256, 128});
      auto b2 = mx::zeros({128});
      auto W3 = mx::random::normal({128, 10});
      auto b3 = mx::zeros({10});
      mx::eval(x, W1, b1, W2, b2, W3, b3);

      entries.push_back({"mlx", measure(iters, [&] {
                            auto h1 = mx::sigmoid(mx::addmm(b1, x, W1));
                            auto h2 = mx::sigmoid(mx::addmm(b2, h1, W2));
                            auto out = mx::addmm(b3, h2, W3);
                            mx::eval(out);
                          })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      long b = batch;
      auto x = torch::randn({b, 784}, dev);
      auto W1 = torch::randn({784, 256}, dev);
      auto b1 = torch::zeros({256}, dev);
      auto W2 = torch::randn({256, 128}, dev);
      auto b2 = torch::zeros({128}, dev);
      auto W3 = torch::randn({128, 10}, dev);
      auto b3 = torch::zeros({10}, dev);

      entries.push_back({"torch", measure(iters, [&] {
                            auto h1 = torch::sigmoid(torch::addmm(b1, x, W1));
                            auto h2 = torch::sigmoid(torch::addmm(b2, h1, W2));
                            auto out = torch::addmm(b3, h2, W3);
                            torch::mps::synchronize();
                          })});
    }
#endif

    auto group = BenchGroup{
        std::format("mlp inference (batch={})", batch), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  bool csv = has_csv_flag(argc, argv);
  std::vector<BenchGroup> groups;
  bench_mlp(groups, csv);
  if (csv) print_csv(groups);
}
