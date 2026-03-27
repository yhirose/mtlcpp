#include <silarray.h>

#include "../bench_common.h"

#ifdef BENCH_HAS_GGML
#include "../bench_ggml.h"
#endif

#ifdef BENCH_HAS_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

#ifdef BENCH_HAS_LIBTORCH
#include <torch/torch.h>
#endif

// Single transformer block: LayerNorm -> Self-Attention -> Residual
//                         -> LayerNorm -> FFN (ReLU) -> Residual

void bench_transformer(std::vector<BenchGroup>& groups, bool csv) {
  if (!csv) print_section("transformer block (single-head)");

  struct Config {
    size_t seq;
    size_t d_model;
    size_t d_ff;
    size_t iters;
  };

  Config configs[] = {
      {256, 512,  2048, 20},    // medium
      {256, 768,  3072, 15},    // BERT-base scale
      {256, 1024, 4096, 10},    // GPT-2 medium scale
      {512, 1024, 4096, 5},     // long sequence
  };

  for (auto& [seq, d_model, d_ff, iters] : configs) {
    std::vector<BenchEntry> entries;
    float scale = 1.0f / sqrtf(static_cast<float>(d_model));

    // --- sil ---
    {
      auto x = sil::random({seq, d_model});
      auto Wq = sil::random({d_model, d_model}) * scale;
      auto bq = sil::zeros<float>({d_model});
      auto Wk = sil::random({d_model, d_model}) * scale;
      auto bk = sil::zeros<float>({d_model});
      auto Wv = sil::random({d_model, d_model}) * scale;
      auto bv = sil::zeros<float>({d_model});
      auto Wo = sil::random({d_model, d_model}) * scale;
      auto bo = sil::zeros<float>({d_model});
      auto W1 = sil::random({d_model, d_ff}) * (1.0f / sqrtf(static_cast<float>(d_model)));
      auto b1 = sil::zeros<float>({d_ff});
      auto W2 = sil::random({d_ff, d_model}) * (1.0f / sqrtf(static_cast<float>(d_ff)));
      auto b2 = sil::zeros<float>({d_model});
      auto gamma1 = sil::ones<float>({d_model});
      auto beta1 = sil::zeros<float>({d_model});
      auto gamma2 = sil::ones<float>({d_model});
      auto beta2 = sil::zeros<float>({d_model});
      sil::synchronize();

      auto transformer_fn = [&](const char* name) {
        entries.push_back({name, measure(iters, [&] {
          auto h = x.layer_norm(gamma1, beta1);
          auto Q = h.linear(Wq, bq);
          auto K = h.linear(Wk, bk);
          auto V = h.linear(Wv, bv);
          auto scores = Q.dot(K.transpose()) * scale;
          auto attn = scores.softmax();
          auto context = attn.dot(V);
          auto attn_out = context.linear(Wo, bo);
          auto r1 = x + attn_out;
          auto h2 = r1.layer_norm(gamma2, beta2);
          auto ff = h2.linear(W1, b1).relu();
          auto out = r1 + ff.linear(W2, b2);
          sil::synchronize();
        })});
      };

      transformer_fn("sil-gpu");
      if (d_model <= 768) {
        sil::use_cpu();
        transformer_fn("sil-cpu");
        sil::use_mps();
      }
    }

#ifdef BENCH_HAS_GGML
    {
      // Tensor shapes in ggml: new_tensor_2d(cols, rows)
      // ggml_mul_mat(A, B): A is (K, N), B is (K, M) -> result (N, M)
      GgmlInputs inputs(17);
      auto* gx      = inputs.new_tensor_2d(d_model, seq);
      auto* gWq     = inputs.new_tensor_2d(d_model, d_model);
      auto* gbq     = inputs.new_tensor_1d(d_model);
      auto* gWk     = inputs.new_tensor_2d(d_model, d_model);
      auto* gbk     = inputs.new_tensor_1d(d_model);
      auto* gWv     = inputs.new_tensor_2d(d_model, d_model);
      auto* gbv     = inputs.new_tensor_1d(d_model);
      auto* gWo     = inputs.new_tensor_2d(d_model, d_model);
      auto* gbo     = inputs.new_tensor_1d(d_model);
      auto* gW1     = inputs.new_tensor_2d(d_model, d_ff);
      auto* gb1     = inputs.new_tensor_1d(d_ff);
      auto* gW2     = inputs.new_tensor_2d(d_ff, d_model);
      auto* gb2     = inputs.new_tensor_1d(d_model);
      auto* ggamma1 = inputs.new_tensor_1d(d_model);
      auto* gbeta1  = inputs.new_tensor_1d(d_model);
      auto* ggamma2 = inputs.new_tensor_1d(d_model);
      auto* gbeta2  = inputs.new_tensor_1d(d_model);

      // Fill with ones (content doesn't matter for benchmarking)
      size_t max_sz = std::max(d_model * d_model, d_ff * d_model);
      max_sz = std::max(max_sz, seq * d_model);
      std::vector<float> ones(max_sz, 1.0f);
      inputs.alloc_and_set(gx, ones.data());
      inputs.alloc_and_set(gWq, ones.data());
      inputs.alloc_and_set(gbq, ones.data());
      inputs.alloc_and_set(gWk, ones.data());
      inputs.alloc_and_set(gbk, ones.data());
      inputs.alloc_and_set(gWv, ones.data());
      inputs.alloc_and_set(gbv, ones.data());
      inputs.alloc_and_set(gWo, ones.data());
      inputs.alloc_and_set(gbo, ones.data());
      inputs.alloc_and_set(gW1, ones.data());
      inputs.alloc_and_set(gb1, ones.data());
      inputs.alloc_and_set(gW2, ones.data());
      inputs.alloc_and_set(gb2, ones.data());
      inputs.alloc_and_set(ggamma1, ones.data());
      inputs.alloc_and_set(gbeta1, ones.data());
      inputs.alloc_and_set(ggamma2, ones.data());
      inputs.alloc_and_set(gbeta2, ones.data());

      entries.push_back({"ggml", measure(iters, [&] {
        auto* ctx_g = ggml_graph_ctx(50);

        // LayerNorm 1: norm then mul gamma + add beta
        auto* normed1 = ggml_norm(ctx_g, gx, 1e-5f);
        auto* ln1 = ggml_add(ctx_g, ggml_mul(ctx_g, normed1, ggamma1), gbeta1);

        // Self-Attention: Q, K, V projections
        auto* Q = ggml_add(ctx_g, ggml_mul_mat(ctx_g, gWq, ln1), gbq);  // (d_model, seq)
        auto* K = ggml_add(ctx_g, ggml_mul_mat(ctx_g, gWk, ln1), gbk);  // (d_model, seq)
        auto* V = ggml_add(ctx_g, ggml_mul_mat(ctx_g, gWv, ln1), gbv);  // (d_model, seq)

        // Attention scores: ggml_mul_mat(K, Q) = Q^T @ K -> (seq, seq)
        auto* scores = ggml_scale(ctx_g, ggml_mul_mat(ctx_g, K, Q), scale);
        auto* attn = ggml_soft_max(ctx_g, scores);

        // Context: attn @ V -> need ggml_mul_mat(V, attn_transposed)
        // attn is (seq, seq), V is (d_model, seq)
        // We want attn @ V which is (seq, seq) @ (seq, d_model) = (seq, d_model)
        // ggml_mul_mat(A, B) = B @ A^T where shapes: A(K,N), B(K,M) -> (N,M)
        // V^T has shape (seq, d_model), so ggml_mul_mat(V_cont_transposed, attn)
        // Simpler: transpose attn to (seq, seq) and use V
        // Actually: ggml_mul_mat(V, ggml_cont(ggml_transpose(attn)))
        //   V is (d_model, seq), attn^T is (seq, seq) -> result (d_model, seq) -- wrong dims
        // Let's think again:
        // We want context = attn @ V where attn(seq,seq) V(seq, d_model) -> (seq, d_model)
        // In ggml col-major: attn is stored as (seq, seq), V as (d_model, seq)
        // ggml_mul_mat(A, B): takes inner dim from A's rows and B's rows
        // ggml_mul_mat(V_transposed, attn) where V_transposed = (seq, d_model)
        //   -> B=attn(seq, seq), A=V_t(seq, d_model) -> result (d_model, seq) = context in col-major
        auto* V_t = ggml_cont(ctx_g, ggml_transpose(ctx_g, V));
        auto* context = ggml_mul_mat(ctx_g, V_t, attn);  // (d_model, seq)

        // Output projection
        auto* attn_out = ggml_add(ctx_g, ggml_mul_mat(ctx_g, gWo, context), gbo);

        // Residual 1
        auto* r1 = ggml_add(ctx_g, gx, attn_out);

        // LayerNorm 2
        auto* normed2 = ggml_norm(ctx_g, r1, 1e-5f);
        auto* ln2 = ggml_add(ctx_g, ggml_mul(ctx_g, normed2, ggamma2), gbeta2);

        // FFN: relu(ln2 @ W1 + b1) @ W2 + b2
        auto* ff = ggml_relu(ctx_g, ggml_add(ctx_g, ggml_mul_mat(ctx_g, gW1, ln2), gb1));
        auto* ffn_out = ggml_add(ctx_g, ggml_mul_mat(ctx_g, gW2, ff), gb2);

        // Residual 2
        auto* out = ggml_add(ctx_g, r1, ffn_out);

        auto* gf = ggml_new_graph(ctx_g);
        ggml_build_forward_expand(gf, out);
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
  auto mode = parse_output_mode(argc, argv);
  bool csv = (mode != OutputMode::bar);
  std::vector<BenchGroup> groups;
#ifdef BENCH_HAS_GGML
  ggml_metal_backend();  // warm up once (logs suppressed internally)
#endif
  bench_transformer(groups, csv);
  if (mode == OutputMode::csv) print_csv(groups);
  if (mode == OutputMode::table) print_table(groups, "Transformer", "Single transformer block (multi-head self-attention + FFN) inference at various sequence lengths and model dimensions");
}
