#include <silarray.h>

#include "../bench_common.h"
#include "mnist_data.h"

#ifdef BENCH_HAS_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

#ifdef BENCH_HAS_LIBTORCH
#include <torch/torch.h>
#endif

// MNIST Autoencoder: 784 -> 512 -> 256 -> 64 -> 256 -> 512 -> 784
// GPU-favorable: large matrices, all linear+sigmoid, no conv2d.
// Training: 1 epoch (600 batches x 100 images), SGD with MSE loss.
// Inference: encode+decode 10000 test images.

static const char* kTrainImages = "../test/train-images-idx3-ubyte";
static const char* kTrainLabels = "../test/train-labels-idx1-ubyte";
static const char* kTestImages  = "../test/t10k-images-idx3-ubyte";
static const char* kTestLabels  = "../test/t10k-labels-idx1-ubyte";

// Layer sizes: 784, 512, 256, 64, 256, 512, 784
static constexpr size_t L[] = {784, 512, 256, 64, 256, 512, 784};
static constexpr size_t N_LAYERS = 6;  // number of weight matrices

void bench_train(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("autoencoder training (1 epoch, batch=100)");

  mnist_data train;
  if (!train.load(kTrainImages, kTrainLabels)) return;

  constexpr size_t batch = 100;
  float lr = 0.01f;

  std::vector<BenchEntry> entries;

  // --- sil ---
  {
    auto run_sil = [&](const char* name) {
      // Initialize weights
      sil::array<float> W[N_LAYERS], b[N_LAYERS];
      for (size_t l = 0; l < N_LAYERS; l++) {
        W[l] = sil::random({L[l], L[l + 1]}) * (2.0f / sqrtf(float(L[l])));
        b[l] = sil::zeros<float>({L[l + 1]});
      }

      entries.push_back({name, measure(3, [&] {
        for (size_t i = 0; i + batch <= train.count; i += batch) {
          auto x = sil::array<float>({batch, 784}, &train.images[i * 784]);

          // Forward: store activations for backward
          sil::array<float> net[N_LAYERS], out[N_LAYERS];
          auto h = x;
          for (size_t l = 0; l < N_LAYERS; l++) {
            net[l] = h.linear(W[l], b[l]);
            out[l] = net[l].sigmoid();
            h = out[l];
          }

          // MSE loss gradient: d/d(out) = 2*(out - target) / n
          auto dout = (2.0f * (h - x)) / float(batch * 784);

          // Backward through all layers
          for (int l = N_LAYERS - 1; l >= 0; l--) {
            dout = net[l].sigmoid_backward(dout);
            auto& input = (l > 0) ? out[l - 1] : x;
            auto dW = input.transpose().dot(dout);
            auto db = dout.sum(0);
            if (l > 0) dout = dout.dot(W[l].transpose());
            W[l] -= dW * lr;
            b[l] -= db * lr;
          }
        }
        sil::synchronize();
      })});
    };

    run_sil("sil-gpu");
    sil::use_cpu();
    run_sil("sil-cpu");
    sil::use_mps();
  }

#ifdef BENCH_HAS_MLX
  {
    int d[] = {784, 512, 256, 64, 256, 512, 784};
    std::vector<mx::array> mW(N_LAYERS, mx::array(0.f)), mb(N_LAYERS, mx::array(0.f));
    for (size_t l = 0; l < N_LAYERS; l++) {
      mW[l] = mx::random::normal({d[l], d[l + 1]}) * (2.0f / sqrtf(float(d[l])));
      mb[l] = mx::zeros({d[l + 1]});
    }
    for (size_t l = 0; l < N_LAYERS; l++) mx::eval(mW[l], mb[l]);

    std::vector<mx::array> batch_x;
    for (size_t i = 0; i + batch <= train.count; i += batch)
      batch_x.push_back(mx::array(&train.images[i * 784], {int(batch), 784}));
    for (auto& a : batch_x) mx::eval(a);

    entries.push_back({"mlx", measure(3, [&] {
      for (auto& x : batch_x) {
        std::vector<mx::array> mnet(N_LAYERS, mx::array(0.f)), mout(N_LAYERS, mx::array(0.f));
        auto h = x;
        for (size_t l = 0; l < N_LAYERS; l++) {
          mnet[l] = mx::addmm(mb[l], h, mW[l]);
          mout[l] = mx::sigmoid(mnet[l]);
          h = mout[l];
        }

        auto dout = mx::multiply(mx::subtract(h, x), mx::array(2.0f / (batch * 784)));

        for (int l = N_LAYERS - 1; l >= 0; l--) {
          auto s = mx::sigmoid(mnet[l]);
          dout = mx::multiply(dout, mx::multiply(s, mx::subtract(mx::array(1.0f), s)));
          auto& input = (l > 0) ? mout[l - 1] : x;
          auto dW = mx::matmul(mx::transpose(input), dout);
          auto db = mx::sum(dout, 0);
          if (l > 0) dout = mx::matmul(dout, mx::transpose(mW[l]));
          auto mlr = mx::array(lr);
          mW[l] = mx::subtract(mW[l], mx::multiply(dW, mlr));
          mb[l] = mx::subtract(mb[l], mx::multiply(db, mlr));
        }
        mx::eval(mW[0], mb[0], mW[1], mb[1], mW[2], mb[2],
                 mW[3], mb[3], mW[4], mb[4], mW[5], mb[5]);
      }
    })});
  }
#endif

#ifdef BENCH_HAS_LIBTORCH
  if (torch::mps::is_available()) {
    auto dev = torch::kMPS;
    long d[] = {784, 512, 256, 64, 256, 512, 784};
    torch::Tensor tW[N_LAYERS], tb[N_LAYERS];
    for (size_t l = 0; l < N_LAYERS; l++) {
      tW[l] = torch::randn({d[l], d[l + 1]}, dev) * (2.0f / sqrtf(float(d[l])));
      tb[l] = torch::zeros({d[l + 1]}, dev);
    }

    std::vector<torch::Tensor> batch_x;
    for (size_t i = 0; i + batch <= train.count; i += batch)
      batch_x.push_back(torch::from_blob(
          const_cast<float*>(&train.images[i * 784]),
          {long(batch), 784}).to(dev));

    entries.push_back({"torch", measure(3, [&] {
      for (auto& x : batch_x) {
        torch::Tensor tnet[N_LAYERS], tout[N_LAYERS];
        auto h = x;
        for (size_t l = 0; l < N_LAYERS; l++) {
          tnet[l] = torch::addmm(tb[l], h, tW[l]);
          tout[l] = torch::sigmoid(tnet[l]);
          h = tout[l];
        }

        auto dout = (h - x) * (2.0f / (batch * 784));

        for (int l = N_LAYERS - 1; l >= 0; l--) {
          auto s = torch::sigmoid(tnet[l]);
          dout = dout * (s * (1.0f - s));
          auto& input = (l > 0) ? tout[l - 1] : x;
          auto dW = input.t().mm(dout);
          auto db = dout.sum(0);
          if (l > 0) dout = dout.mm(tW[l].t());
          tW[l] -= dW * lr;
          tb[l] -= db * lr;
        }
        torch::mps::synchronize();
      }
    })});
  }
#endif

  auto group = BenchGroup{"train 1 epoch (60000 images)", std::move(entries)};
  if (!csv) print_group(group);
  groups.push_back(std::move(group));
}

void bench_inference(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("autoencoder inference (10000 images)");

  mnist_data test;
  if (!test.load(kTestImages, kTestLabels)) return;

  std::vector<BenchEntry> entries;

  // --- sil ---
  {
    auto run_sil = [&](const char* name) {
      sil::array<float> W[N_LAYERS], b[N_LAYERS];
      for (size_t l = 0; l < N_LAYERS; l++) {
        W[l] = sil::random({L[l], L[l + 1]});
        b[l] = sil::zeros<float>({L[l + 1]});
      }
      auto x = sil::array<float>({test.count, 784}, test.images.data());
      sil::synchronize();

      entries.push_back({name, measure(10, [&] {
        auto h = x;
        for (size_t l = 0; l < N_LAYERS; l++)
          h = h.linear(W[l], b[l]).sigmoid();
        sil::synchronize();
      })});
    };

    run_sil("sil-gpu");
    sil::use_cpu();
    run_sil("sil-cpu");
    sil::use_mps();
  }

#ifdef BENCH_HAS_MLX
  {
    int d[] = {784, 512, 256, 64, 256, 512, 784};
    std::vector<mx::array> mW(N_LAYERS, mx::array(0.f)), mb(N_LAYERS, mx::array(0.f));
    for (size_t l = 0; l < N_LAYERS; l++) {
      mW[l] = mx::random::normal({d[l], d[l + 1]});
      mb[l] = mx::zeros({d[l + 1]});
    }
    auto x = mx::array(test.images.data(), {int(test.count), 784});
    for (size_t l = 0; l < N_LAYERS; l++) mx::eval(mW[l], mb[l]);
    mx::eval(x);

    entries.push_back({"mlx", measure(10, [&] {
      auto h = x;
      for (size_t l = 0; l < N_LAYERS; l++)
        h = mx::sigmoid(mx::addmm(mb[l], h, mW[l]));
      mx::eval(h);
    })});
  }
#endif

#ifdef BENCH_HAS_LIBTORCH
  if (torch::mps::is_available()) {
    auto dev = torch::kMPS;
    long d[] = {784, 512, 256, 64, 256, 512, 784};
    torch::Tensor tW[N_LAYERS], tb[N_LAYERS];
    for (size_t l = 0; l < N_LAYERS; l++) {
      tW[l] = torch::randn({d[l], d[l + 1]}, dev);
      tb[l] = torch::zeros({d[l + 1]}, dev);
    }
    auto x = torch::from_blob(const_cast<float*>(test.images.data()),
                               {long(test.count), 784}).to(dev);

    entries.push_back({"torch", measure(10, [&] {
      auto h = x;
      for (size_t l = 0; l < N_LAYERS; l++)
        h = torch::sigmoid(torch::addmm(tb[l], h, tW[l]));
      torch::mps::synchronize();
    })});
  }
#endif

  auto group = BenchGroup{"inference (10000 images)", std::move(entries)};
  if (!csv) print_group(group);
  groups.push_back(std::move(group));
}

int main(int argc, const char** argv) {
  auto mode = parse_output_mode(argc, argv);
  bool csv = (mode != OutputMode::bar);
  std::vector<BenchGroup> groups;

  bench_train(groups, csv);
  bench_inference(groups, csv);

  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "MNIST Autoencoder",
      "784->512->256->64->256->512->784 (sigmoid, MSE loss, SGD). Training: 1 epoch, batch=100. Inference: 10000 images.");
}
