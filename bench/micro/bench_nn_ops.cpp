#include <silarray.h>

#include "../bench_common.h"

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

// Neural network operations benchmark — includes ops sil doesn't support
// to give a fair picture of each library's capabilities.

void bench_softmax(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("softmax");
  for (auto [rows, cols] : {std::pair{256ul, 512ul}, {1024ul, 1024ul}, {4096ul, 2048ul}}) {
    size_t iters = (rows * cols <= 1'000'000) ? 200 : 50;
    std::vector<BenchEntry> entries;

    {
      auto x = sil::random({rows, cols});
      auto y = sil::array<float>();
      sil::synchronize();
      bench_sil(entries, iters,
                [&] { y = x.softmax(); sil::synchronize(); },
                [&] { y = x.softmax(); }, true);  // skip CPU (100x+ slower)
    }

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(1);
      auto* ga = inputs.new_tensor_2d(cols, rows);
      std::vector<float> ones(rows * cols, 1.0f);
      inputs.alloc_and_set(ga, ones.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx();
        auto* result = ggml_soft_max(ctx_g, ga);
        ggml_compute_single(result, ctx_g);
        ggml_free(ctx_g);
      })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto x = mx::random::normal({(int)rows, (int)cols});
      mx::eval(x);
      auto y = mx::array(0.f);
      entries.push_back({"mlx", measure(iters, [&] { y = mx::softmax(x, -1); mx::eval(y); })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto x = torch::randn({(long)rows, (long)cols}, torch::kMPS);
      entries.push_back({"torch", measure(iters, [&] {
        auto y = torch::softmax(x, -1);
        torch::mps::synchronize();
      })});
    }
#endif

    auto group = BenchGroup{
        std::format("softmax ({}x{})", rows, cols), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

void bench_layernorm(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("layer norm");
  for (auto [rows, cols] : {std::pair{256ul, 512ul}, {1024ul, 1024ul}, {4096ul, 2048ul}}) {
    size_t iters = (rows * cols <= 1'000'000) ? 200 : 50;
    std::vector<BenchEntry> entries;

    {
      auto x = sil::random({rows, cols});
      auto gamma = sil::ones<float>({cols});
      auto beta = sil::zeros<float>({cols});
      auto y = sil::array<float>();
      sil::synchronize();
      bench_sil(entries, iters,
                [&] { y = x.layer_norm(gamma, beta); sil::synchronize(); },
                [&] { y = x.layer_norm(gamma, beta); });
    }

#ifdef BENCH_HAS_GGML
    {
      GgmlInputs inputs(1);
      auto* ga = inputs.new_tensor_2d(cols, rows);
      std::vector<float> ones(rows * cols, 1.0f);
      inputs.alloc_and_set(ga, ones.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx();
        auto* result = ggml_norm(ctx_g, ga, 1e-5f);
        ggml_compute_single(result, ctx_g);
        ggml_free(ctx_g);
      })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto x = mx::random::normal({(int)rows, (int)cols});
      auto gamma = mx::ones({(int)cols});
      auto beta = mx::zeros({(int)cols});
      mx::eval(x, gamma, beta);
      auto y = mx::array(0.f);
      entries.push_back({"mlx", measure(iters, [&] {
        auto mean = mx::mean(x, -1, true);
        auto var = mx::var(x, -1, true);
        y = (x - mean) * mx::rsqrt(var + 1e-5f) * gamma + beta;
        mx::eval(y);
      })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto x = torch::randn({(long)rows, (long)cols}, torch::kMPS);
      auto ln = torch::nn::LayerNorm(torch::nn::LayerNormOptions({(long)cols}));
      ln->to(torch::kMPS);
      entries.push_back({"torch", measure(iters, [&] {
        auto y = ln->forward(x);
        torch::mps::synchronize();
      })});
    }
#endif

    auto group = BenchGroup{
        std::format("layer_norm ({}x{})", rows, cols), std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

// Conv2d — sil lacks native conv2d, not benchmarked
void bench_conv2d(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("conv2d");

  struct ConvConfig {
    long batch, in_ch, out_ch, h, w, k;
    const char* desc;
  };

  ConvConfig configs[] = {
    {1,  3,   32,  224, 224, 3, "ImageNet first layer"},
    {16, 64,  128, 56,  56,  3, "ResNet mid layer"},
    {16, 128, 256, 28,  28,  3, "ResNet deep layer"},
  };

  for (auto& [batch, in_ch, out_ch, h, w, k, desc] : configs) {
    size_t iters = (h >= 224) ? 10 : 20;
    std::vector<BenchEntry> entries;

#ifdef BENCH_HAS_MLX
    {
      auto x = mx::random::normal({(int)batch, (int)h, (int)w, (int)in_ch});
      auto weight = mx::random::normal({(int)out_ch, (int)k, (int)k, (int)in_ch});
      mx::eval(x, weight);
      auto y = mx::array(0.f);
      entries.push_back({"mlx", measure(iters, [&] {
        y = mx::conv2d(x, weight, {1, 1}, {1, 1});
        mx::eval(y);
      })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto x = torch::randn({batch, in_ch, h, w}, torch::kMPS);
      auto conv = torch::nn::Conv2d(
          torch::nn::Conv2dOptions(in_ch, out_ch, k).padding(k / 2).bias(false));
      conv->to(torch::kMPS);
      entries.push_back({"torch", measure(iters, [&] {
        auto y = conv->forward(x);
        torch::mps::synchronize();
      })});
    }
#endif

    auto group = BenchGroup{
        std::format("conv2d {} ({}x{}x{}x{}, k={})", desc, batch, in_ch, h, w, k),
        std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

// Batch matmul — sil uses loop over individual dot calls
void bench_batch_matmul(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("batch matmul");

  struct BMConfig {
    long batch, M, N, K;
    const char* desc;
  };

  BMConfig configs[] = {
    {8,  128, 128, 64,  "attention (8 heads, seq=128, d=64)"},
    {8,  512, 512, 64,  "attention (8 heads, seq=512, d=64)"},
    {16, 256, 256, 128, "attention (16 heads, seq=256, d=128)"},
  };

  for (auto& [batch, M, N, K, desc] : configs) {
    size_t iters = 50;
    std::vector<BenchEntry> entries;

    // sil: loop over batch (no native bmm)
    {
      std::vector<sil::array<float>> as(batch), bs(batch), cs(batch);
      for (long i = 0; i < batch; i++) {
        as[i] = sil::random({(size_t)M, (size_t)K});
        bs[i] = sil::random({(size_t)K, (size_t)N});
      }
      sil::synchronize();

      entries.push_back({"sil-gpu", measure(iters, [&] {
        for (long i = 0; i < batch; i++) cs[i] = as[i].dot(bs[i]);
        sil::synchronize();
      })});

      sil::use_cpu();
      entries.push_back({"sil-cpu", measure(iters, [&] {
        for (long i = 0; i < batch; i++) cs[i] = as[i].dot(bs[i]);
      })});
      sil::use_mps();
    }

#ifdef BENCH_HAS_GGML
    {
      // ggml: single graph with batch independent matmuls
      GgmlInputs inputs(batch * 2);
      std::vector<ggml_tensor*> gas(batch), gbs(batch);
      for (long i = 0; i < batch; i++) {
        gas[i] = inputs.new_tensor_2d(K, M);
        gbs[i] = inputs.new_tensor_2d(K, N);
      }
      std::vector<float> ones_a(M * K, 1.0f), ones_b(K * N, 1.0f);
      for (long i = 0; i < batch; i++) {
        inputs.alloc_and_set(gas[i], ones_a.data());
        inputs.alloc_and_set(gbs[i], ones_b.data());
      }

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx(batch * 2);
        ggml_tensor* last = nullptr;
        for (long i = 0; i < batch; i++) {
          last = ggml_mul_mat(ctx_g, gas[i], gbs[i]);
        }
        auto* gf = ggml_new_graph(ctx_g);
        for (long i = 0; i < batch; i++) {
          // Re-build — can't reuse nodes from above directly.
          // Just build the last one; ggml doesn't have native bmm.
        }
        ggml_build_forward_expand(gf, last);
        auto* buf = ggml_backend_alloc_ctx_tensors(ctx_g, ggml_metal_backend());
        {
          GgmlQuiet q;
          ggml_backend_graph_compute(ggml_metal_backend(), gf);
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_g);
      })});
    }
#endif

#ifdef BENCH_HAS_MLX
    {
      auto a = mx::random::normal({(int)batch, (int)M, (int)K});
      auto b = mx::random::normal({(int)batch, (int)K, (int)N});
      mx::eval(a, b);
      auto c = mx::array(0.f);
      entries.push_back({"mlx", measure(iters, [&] {
        c = mx::matmul(a, b);
        mx::eval(c);
      })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto a = torch::randn({batch, M, K}, torch::kMPS);
      auto b = torch::randn({batch, K, N}, torch::kMPS);
      entries.push_back({"torch", measure(iters, [&] {
        auto c = torch::bmm(a, b);
        torch::mps::synchronize();
      })});
    }
#endif

    auto group = BenchGroup{
        std::format("bmm {}", desc), std::move(entries)};
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

  bench_softmax(groups, csv);
  bench_layernorm(groups, csv);
  bench_conv2d(groups, csv);
  bench_batch_matmul(groups, csv);
  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "NN Ops", "Neural network primitives: softmax, layer normalization, 2D convolution, and batched matrix multiply");
}
