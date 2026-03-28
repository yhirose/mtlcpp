#include <silarray.h>

#include "bench_common.h"

#ifdef BENCH_HAS_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

#ifdef BENCH_HAS_LIBTORCH
#include <torch/torch.h>
#endif

// Single transformer block: LayerNorm -> Self-Attention -> Residual
//                         -> LayerNorm -> FFN (ReLU) -> Residual
// Tests matmul-heavy attention pattern, manual LayerNorm, and manual ReLU.

// --- Manual LayerNorm for sil ---
// x: (seq, d_model), gamma/beta: (d_model,)
inline sil::array<float> layer_norm(const sil::array<float>& x,
                                    const sil::array<float>& gamma,
                                    const sil::array<float>& beta,
                                    size_t seq) {
  constexpr float eps = 1e-5f;
  auto mu = x.mean(1);
  mu.reshape({seq, 1});
  auto diff = x - mu;
  auto var = (diff * diff).mean(1);
  var.reshape({seq, 1});
  auto normed = diff / (var + sil::array<float>({seq, 1}, eps)).pow(sil::array<float>(0.5f));
  return normed * gamma + beta;
}

// --- Manual ReLU for sil ---
// where(cond, scalar, scalar) * x — two passes but works with existing ops
inline sil::array<float> relu(const sil::array<float>& x) {
  auto mask = sil::where(x > sil::array<float>(0.0f), 1.0f, 0.0f);
  return mask * x;
}

void bench_transformer(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("transformer block (single-head)");

  struct Config {
    size_t seq;
    size_t d_model;
    size_t d_ff;
    size_t iters;
  };

  Config configs[] = {
      {32, 64, 256, 200},
      {64, 128, 512, 100},
      {128, 256, 1024, 50},
  };

  for (auto& [seq, d_model, d_ff, iters] : configs) {
    std::vector<BenchEntry> entries;
    float scale = 1.0f / sqrtf(static_cast<float>(d_model));

    // --- sil ---
    {
      auto x = sil::random({seq, d_model});

      // Attention weights
      auto Wq = sil::random({d_model, d_model}) * scale;
      auto bq = sil::zeros<float>({d_model});
      auto Wk = sil::random({d_model, d_model}) * scale;
      auto bk = sil::zeros<float>({d_model});
      auto Wv = sil::random({d_model, d_model}) * scale;
      auto bv = sil::zeros<float>({d_model});
      auto Wo = sil::random({d_model, d_model}) * scale;
      auto bo = sil::zeros<float>({d_model});

      // FFN weights
      auto W1 = sil::random({d_model, d_ff}) * (1.0f / sqrtf(static_cast<float>(d_model)));
      auto b1 = sil::zeros<float>({d_ff});
      auto W2 = sil::random({d_ff, d_model}) * (1.0f / sqrtf(static_cast<float>(d_ff)));
      auto b2 = sil::zeros<float>({d_model});

      // LayerNorm params
      auto gamma1 = sil::ones<float>({d_model});
      auto beta1 = sil::zeros<float>({d_model});
      auto gamma2 = sil::ones<float>({d_model});
      auto beta2 = sil::zeros<float>({d_model});

      entries.push_back({"sil", measure(iters, [&] {
        // LayerNorm 1
        auto h = layer_norm(x, gamma1, beta1, seq);

        // Self-Attention
        auto Q = h.linear(Wq, bq);
        auto K = h.linear(Wk, bk);
        auto V = h.linear(Wv, bv);

        auto scores = Q.dot(K.transpose()) * scale;
        auto attn = scores.softmax();
        auto context = attn.dot(V);
        auto attn_out = context.linear(Wo, bo);

        // Residual 1
        auto r1 = x + attn_out;

        // LayerNorm 2
        auto h2 = layer_norm(r1, gamma2, beta2, seq);

        // FFN with ReLU
        auto ff = relu(h2.linear(W1, b1));
        auto ffn_out = ff.linear(W2, b2);

        // Residual 2
        auto out = r1 + ffn_out;

        sil::synchronize();
      })});
    }

#ifdef BENCH_HAS_MLX
    {
      int s = static_cast<int>(seq);
      int dm = static_cast<int>(d_model);
      int df = static_cast<int>(d_ff);

      auto x = mx::random::normal({s, dm});

      auto Wq = mx::random::normal({dm, dm}) * scale;
      auto bq = mx::zeros({dm});
      auto Wk = mx::random::normal({dm, dm}) * scale;
      auto bk = mx::zeros({dm});
      auto Wv = mx::random::normal({dm, dm}) * scale;
      auto bv = mx::zeros({dm});
      auto Wo = mx::random::normal({dm, dm}) * scale;
      auto bo = mx::zeros({dm});

      auto W1 = mx::random::normal({dm, df}) * (1.0f / sqrtf(static_cast<float>(d_model)));
      auto b1 = mx::zeros({df});
      auto W2 = mx::random::normal({df, dm}) * (1.0f / sqrtf(static_cast<float>(d_ff)));
      auto b2 = mx::zeros({dm});

      auto gamma1 = mx::ones({dm});
      auto beta1 = mx::zeros({dm});
      auto gamma2 = mx::ones({dm});
      auto beta2 = mx::zeros({dm});

      mx::eval(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, W1, b1, W2, b2,
               gamma1, beta1, gamma2, beta2);

      auto mlx_layer_norm = [&](const mx::array& x, const mx::array& gamma,
                                const mx::array& beta) {
        constexpr float eps = 1e-5f;
        auto mu = mx::mean(x, /* axis= */ 1, /* keepdims= */ true);
        auto diff = mx::subtract(x, mu);
        auto var = mx::mean(mx::multiply(diff, diff), 1, true);
        auto normed = mx::multiply(diff, mx::rsqrt(mx::add(var, mx::array(eps))));
        return mx::add(mx::multiply(normed, gamma), beta);
      };

      entries.push_back({"mlx", measure(iters, [&] {
        auto h = mlx_layer_norm(x, gamma1, beta1);

        auto Q = mx::addmm(bq, h, Wq);
        auto K = mx::addmm(bk, h, Wk);
        auto V = mx::addmm(bv, h, Wv);

        auto scores = mx::multiply(mx::matmul(Q, mx::transpose(K)), mx::array(scale));
        auto attn = mx::softmax(scores, -1);
        auto context = mx::matmul(attn, V);
        auto attn_out = mx::addmm(bo, context, Wo);

        auto r1 = mx::add(x, attn_out);

        auto h2 = mlx_layer_norm(r1, gamma2, beta2);

        auto ff = mx::maximum(mx::addmm(b1, h2, W1), mx::array(0.0f));
        auto ffn_out = mx::addmm(b2, ff, W2);

        auto out = mx::add(r1, ffn_out);
        mx::eval(out);
      })});
    }
#endif

#ifdef BENCH_HAS_LIBTORCH
    if (torch::mps::is_available()) {
      auto dev = torch::kMPS;
      long s = seq;
      long dm = d_model;
      long df = d_ff;

      auto x = torch::randn({s, dm}, dev);

      auto Wq = torch::randn({dm, dm}, dev) * scale;
      auto bq = torch::zeros({dm}, dev);
      auto Wk = torch::randn({dm, dm}, dev) * scale;
      auto bk = torch::zeros({dm}, dev);
      auto Wv = torch::randn({dm, dm}, dev) * scale;
      auto bv = torch::zeros({dm}, dev);
      auto Wo = torch::randn({dm, dm}, dev) * scale;
      auto bo = torch::zeros({dm}, dev);

      auto W1 = torch::randn({dm, df}, dev) * (1.0f / sqrtf(static_cast<float>(d_model)));
      auto b1 = torch::zeros({df}, dev);
      auto W2 = torch::randn({df, dm}, dev) * (1.0f / sqrtf(static_cast<float>(d_ff)));
      auto b2 = torch::zeros({dm}, dev);

      auto gamma1 = torch::ones({dm}, dev);
      auto beta1 = torch::zeros({dm}, dev);
      auto gamma2 = torch::ones({dm}, dev);
      auto beta2 = torch::zeros({dm}, dev);

      auto torch_layer_norm = [&](const torch::Tensor& x,
                                  const torch::Tensor& gamma,
                                  const torch::Tensor& beta) {
        constexpr float eps = 1e-5f;
        auto mu = x.mean(/* dim= */ 1, /* keepdim= */ true);
        auto diff = x - mu;
        auto var = (diff * diff).mean(1, true);
        auto normed = diff * (var + eps).rsqrt();
        return normed * gamma + beta;
      };

      entries.push_back({"torch", measure(iters, [&] {
        auto h = torch_layer_norm(x, gamma1, beta1);

        auto Q = torch::addmm(bq, h, Wq);
        auto K = torch::addmm(bk, h, Wk);
        auto V = torch::addmm(bv, h, Wv);

        auto scores = torch::mm(Q, K.t()) * scale;
        auto attn = torch::softmax(scores, -1);
        auto context = torch::mm(attn, V);
        auto attn_out = torch::addmm(bo, context, Wo);

        auto r1 = x + attn_out;

        auto h2 = torch_layer_norm(r1, gamma2, beta2);

        auto ff = torch::relu(torch::addmm(b1, h2, W1));
        auto ffn_out = torch::addmm(b2, ff, W2);

        auto out = r1 + ffn_out;
        torch::mps::synchronize();
      })});
    }
#endif

    auto group = BenchGroup{
        std::format("transformer (seq={}, d={})", seq, d_model),
        std::move(entries)};
    if (!csv) print_group(group);
    groups.push_back(std::move(group));
  }
}

int main(int argc, const char** argv) {
  bool csv = has_csv_flag(argc, argv);
  std::vector<BenchGroup> groups;
  bench_transformer(groups, csv);
  if (csv) print_csv(groups);
}
