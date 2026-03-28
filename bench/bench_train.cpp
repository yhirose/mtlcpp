#include <sil.h>

#include "bench_common.h"

#ifdef BENCH_HAS_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

#ifdef BENCH_HAS_LIBTORCH
#include <torch/torch.h>
#endif

// Training step: forward + loss + backward + weight update for a 2-layer MLP.
// The most realistic end-to-end benchmark.

void bench_train(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("training step (784->50->10)");
  for (auto batch : {32ul, 128ul, 512ul}) {
    size_t iters = batch <= 32 ? 200 : 50;
    float lr = 0.1f;

    std::vector<BenchEntry> entries;

    // --- sil ---
    {
      auto W1 = sil::random({784, 50}) * (1.0f / sqrtf(784.0f));
      auto b1 = sil::zeros<float>({50});
      auto W2 = sil::random({50, 10}) * (1.0f / sqrtf(50.0f));
      auto b2 = sil::zeros<float>({10});
      auto x = sil::random({batch, 784});
      auto Y = sil::zeros<float>({batch, 10});
      // One-hot target: class 0 for simplicity
      for (size_t i = 0; i < batch; i++) Y.at({i, 0}) = 1.0f;

      entries.push_back({"sil", measure(iters, [&] {
        // Forward
        auto n1 = x.linear(W1, b1);
        auto o1 = n1.sigmoid();
        auto n2 = o1.linear(W2, b2);
        auto o2 = n2.sigmoid();

        // Loss gradient (MSE derivative)
        auto dout = (2.0f * (o2 - Y)) / static_cast<float>(Y.length());

        // Backward through sigmoid2
        auto s2 = n2.sigmoid();
        dout = dout * (s2 * (1.0f - s2));

        // Backward through linear2
        auto dW2 = o1.transpose().dot(dout);
        auto db2 = dout.sum(0);
        auto dout1 = dout.dot(W2.transpose());

        // Backward through sigmoid1
        auto s1 = n1.sigmoid();
        dout1 = dout1 * (s1 * (1.0f - s1));

        // Backward through linear1
        auto dW1 = x.transpose().dot(dout1);
        auto db1 = dout1.sum(0);

        // Update
        W1 -= dW1 * lr;
        b1 -= db1 * lr;
        W2 -= dW2 * lr;
        b2 -= db2 * lr;

        sil::synchronize();
      })});
    }

#ifdef BENCH_HAS_MLX
    {
      int b = static_cast<int>(batch);
      auto W1 = mx::random::normal({784, 50}) * (1.0f / sqrtf(784.0f));
      auto b1 = mx::zeros({50});
      auto W2 = mx::random::normal({50, 10}) * (1.0f / sqrtf(50.0f));
      auto b2 = mx::zeros({10});
      auto x = mx::random::normal({b, 784});
      auto Y = mx::zeros({b, 10});
      mx::eval(W1, b1, W2, b2, x, Y);

      entries.push_back({"mlx", measure(iters, [&] {
        auto n1 = mx::addmm(b1, x, W1);
        auto o1 = mx::sigmoid(n1);
        auto n2 = mx::addmm(b2, o1, W2);
        auto o2 = mx::sigmoid(n2);

        auto dout = mx::multiply(mx::subtract(o2, Y), mx::array(2.0f / b));

        auto s2 = mx::sigmoid(n2);
        dout = mx::multiply(dout, mx::multiply(s2, mx::subtract(mx::array(1.0f), s2)));

        auto dW2 = mx::matmul(mx::transpose(o1), dout);
        auto db2 = mx::sum(dout, 0);
        auto dout1 = mx::matmul(dout, mx::transpose(W2));

        auto s1 = mx::sigmoid(n1);
        dout1 = mx::multiply(dout1, mx::multiply(s1, mx::subtract(mx::array(1.0f), s1)));

        auto dW1 = mx::matmul(mx::transpose(x), dout1);
        auto db1 = mx::sum(dout1, 0);

        auto mlr = mx::array(lr);
        W1 = mx::subtract(W1, mx::multiply(dW1, mlr));
        b1 = mx::subtract(b1, mx::multiply(db1, mlr));
        W2 = mx::subtract(W2, mx::multiply(dW2, mlr));
        b2 = mx::subtract(b2, mx::multiply(db2, mlr));

        mx::eval(W1, b1, W2, b2);
      })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      long b = batch;
      auto W1 = torch::randn({784, 50}, dev) * (1.0f / sqrtf(784.0f));
      auto b1 = torch::zeros({50}, dev);
      auto W2 = torch::randn({50, 10}, dev) * (1.0f / sqrtf(50.0f));
      auto b2 = torch::zeros({10}, dev);
      auto x = torch::randn({b, 784}, dev);
      auto Y = torch::zeros({b, 10}, dev);

      entries.push_back({"torch", measure(iters, [&] {
        auto n1 = torch::addmm(b1, x, W1);
        auto o1 = torch::sigmoid(n1);
        auto n2 = torch::addmm(b2, o1, W2);
        auto o2 = torch::sigmoid(n2);

        auto dout = (o2 - Y) * (2.0f / b);

        auto s2 = torch::sigmoid(n2);
        dout = dout * (s2 * (1.0f - s2));

        auto dW2 = o1.t().mm(dout);
        auto db2 = dout.sum(0);
        auto dout1 = dout.mm(W2.t());

        auto s1 = torch::sigmoid(n1);
        dout1 = dout1 * (s1 * (1.0f - s1));

        auto dW1 = x.t().mm(dout1);
        auto db1 = dout1.sum(0);

        W1 -= dW1 * lr;
        b1 -= db1 * lr;
        W2 -= dW2 * lr;
        b2 -= db2 * lr;

        torch::mps::synchronize();
      })});
    }
#endif

    auto group = BenchGroup{
        std::format("train step (batch={})", batch), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  bool csv = has_csv_flag(argc, argv);
  std::vector<BenchGroup> groups;
  bench_train(groups, csv);
  if (csv) print_csv(groups);
}
