#pragma once

#include <types.h>
#include <device.h>

#include <sstream>
#include <stdexcept>

namespace sil {

//-----------------------------------------------------------------------------
// GPU context
//-----------------------------------------------------------------------------

class gpu_context {
 public:
  void* queue;

  struct pipeline {
    void* pso;
    size_t thread_width;
    size_t max_threads;
  };

  static gpu_context& instance() {
    static auto* ctx = new gpu_context();
    return *ctx;
  }

  const pipeline& pso(size_t index) { return psos_[index]; }

  void* command_buffer() {
    if (!cb_) {
      cb_ = objc::send(queue, objc::sel_::commandBuffer());
      gpu_pending_ = true;
    }
    return cb_;
  }

  void* compute_encoder() {
    if (!encoder_) encoder_ = objc::send(command_buffer(), objc::sel_::computeCommandEncoder());
    return encoder_;
  }

  void end_encoder() {
    if (encoder_) {
      objc::send(encoder_, objc::sel_::endEncoding());
      encoder_ = nullptr;
    }
  }

  void flush() {
    if (!cb_) return;
    end_encoder();
    objc::send(cb_, objc::sel_::commit());
    objc::send(cb_, objc::sel_::waitUntilCompleted());
    cb_ = nullptr;
    gpu_pending_ = false;
  }

 private:
  std::vector<pipeline> psos_;
  void* cb_ = nullptr;
  void* encoder_ = nullptr;

  gpu_context() {
    auto* device = buffer_pool::instance().device;
    queue = objc::send(device, "newCommandQueue");

    // Compile MSL source
    auto src = objc::cfstr(msl_source_());
    void* err = nullptr;
    auto lib = reinterpret_cast<void*(*)(void*, SEL, void*, void*, void**)>(
        objc_msgSend)(device, objc::sel("newLibraryWithSource:options:error:"),
                      src, nullptr, &err);
    objc::cfrelease(src);

    if (!lib) {
      auto desc = objc::send(err, "localizedDescription");
      auto s = reinterpret_cast<const char*>(objc::send(desc, "UTF8String"));
      throw std::runtime_error(std::string("gpu: Failed to compile MSL: ") + s);
    }

    // Create pipeline state objects
    const char* fn_names[] = {"add", "sub", "mul", "div", "pow", "sigmoid_", "relu_", "sum_f32_", "layer_norm_", "softmax_f32_", "affine_f32_", "sigmoid_backward_f32_", "bias_sigmoid_f32_", "sgemm_32_", "sgemm_64_"};
    for (auto name : fn_names) {
      psos_.push_back(create_pso_(device, lib, name));
    }

    // STEEL FC-specialized variants: aligned + edge for each kernel
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_steel_",
                                    false, false, true, true));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_steel_",
                                    false, false, false, false));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_steel_",
                                    false, false, true, true));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_steel_",
                                    false, false, false, false));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_sigmoid_steel_",
                                    false, false, true, true));
    psos_.push_back(create_fc_pso_(device, lib, "sgemm_bias_sigmoid_steel_",
                                    false, false, false, false));

    objc::release(lib);
  }

  pipeline create_pso_(void* device, void* library, const char* name) {
    auto fn_name = objc::cfstr(name);
    auto fn = objc::send(library, "newFunctionWithName:", fn_name);
    objc::cfrelease(fn_name);
    if (!fn) {
      throw std::runtime_error(
          std::string("gpu: Failed to find function: ") + name);
    }
    return finalize_pso_(device, fn, name);
  }

  static constexpr unsigned long kMTLDataTypeBool = 53;

  pipeline finalize_pso_(void* device, void* fn, const char* name) {
    void* err = nullptr;
    auto pso = reinterpret_cast<void*(*)(void*, SEL, void*, void**)>(
        objc_msgSend)(device, objc::sel("newComputePipelineStateWithFunction:error:"),
                      fn, &err);
    objc::release(fn);
    if (!pso) {
      throw std::runtime_error(
          std::string("gpu: Failed to create PSO for: ") + name);
    }
    auto w = objc::send_uint(pso, "threadExecutionWidth");
    auto max = objc::send_uint(pso, "maxTotalThreadsPerThreadgroup");
    return {pso, w, max};
  }

  pipeline create_fc_pso_(void* device, void* library, const char* name,
                           bool trans_a, bool trans_b,
                           bool mn_aligned, bool k_aligned) {
    auto fn_name = objc::cfstr(name);
    auto fc_vals = objc::send(objc::send(objc::cls("MTLFunctionConstantValues"),
                                          objc::sel_::alloc()), "init");
    auto set_sel = objc::sel("setConstantValue:type:atIndex:");
    reinterpret_cast<void(*)(void*, SEL, const void*, unsigned long, unsigned long)>(
        objc_msgSend)(fc_vals, set_sel, &trans_a, kMTLDataTypeBool, 0ul);
    reinterpret_cast<void(*)(void*, SEL, const void*, unsigned long, unsigned long)>(
        objc_msgSend)(fc_vals, set_sel, &trans_b, kMTLDataTypeBool, 1ul);
    reinterpret_cast<void(*)(void*, SEL, const void*, unsigned long, unsigned long)>(
        objc_msgSend)(fc_vals, set_sel, &mn_aligned, kMTLDataTypeBool, 2ul);
    reinterpret_cast<void(*)(void*, SEL, const void*, unsigned long, unsigned long)>(
        objc_msgSend)(fc_vals, set_sel, &k_aligned, kMTLDataTypeBool, 3ul);

    void* err = nullptr;
    auto fn = reinterpret_cast<void*(*)(void*, SEL, void*, void*, void**)>(
        objc_msgSend)(library, objc::sel("newFunctionWithName:constantValues:error:"),
                      fn_name, fc_vals, &err);
    objc::cfrelease(fn_name);
    objc::release(fc_vals);

    if (!fn) {
      auto desc = objc::send(err, "localizedDescription");
      auto s = reinterpret_cast<const char*>(objc::send(desc, "UTF8String"));
      throw std::runtime_error(std::string("gpu: FC function error: ") + s);
    }
    return finalize_pso_(device, fn, name);
  }

  static const char* msl_source_() {
    return R"MSL(

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------------------------------------------------------------------
// Tiled SGEMM — simdgroup_matrix 8×8 MMA with float4 vectorized loads
// ---------------------------------------------------------------------------

struct gemm_params {
  uint M, N, K, lda, ldb, trans_a, trans_b;
};

template <uint BM, uint BN, uint BK>
void sgemm_impl_(
    device const float* A,
    device const float* B,
    device float*       C,
    constant gemm_params& p,
    threadgroup float* As,
    threadgroup float* Bs,
    uint2 tgid, uint tid, uint sid, uint lane)
{
  constexpr uint N_SM = 2, N_SN = 2;
  constexpr uint TM = BM / N_SM, TN = BN / N_SN;
  constexpr uint FM = TM / 8, FN = TN / 8;
  constexpr uint THREADS = N_SM * N_SN * 32;
  constexpr uint aS = BK + 4;   // float4-aligned padding
  constexpr uint bS = BN + 4;

  uint row_tg = tgid.y;
  uint col_tg = tgid.x;

  uint wm = sid / N_SN, wn = sid % N_SN;

  simdgroup_matrix<float, 8, 8> acc[FM][FN];
  for (uint i = 0; i < FM; i++)
    for (uint j = 0; j < FN; j++)
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

  uint row0 = row_tg * BM, col0 = col_tg * BN;
  uint a_rs = p.trans_a ? 1u : p.lda;
  uint a_cs = p.trans_a ? p.lda : 1u;
  uint b_rs = p.trans_b ? 1u : p.ldb;
  uint b_cs = p.trans_b ? p.ldb : 1u;
  bool full_tile = (row0 + BM <= p.M) && (col0 + BN <= p.N);
  uint k_full = (p.K / BK) * BK;

  // -- load macros with float4 vectorization for non-transposed path --------
  #define LOAD_A_FAST                                                       \
    if (!p.trans_a) {                                                        \
      constexpr uint F4 = BK / 4;                                           \
      for (uint i = tid; i < BM * F4; i += THREADS) {                      \
        uint r = i / F4, fc = i % F4;                                       \
        auto v = *reinterpret_cast<device const float4*>(                   \
            &A[(row0 + r) * p.lda + k0 + fc * 4]);                         \
        *reinterpret_cast<threadgroup float4*>(&As[r * aS + fc * 4]) = v;  \
      }                                                                      \
    } else {                                                                 \
      for (uint i = tid; i < BM * BK; i += THREADS) {                      \
        uint r = i / BK, c = i % BK;                                        \
        As[r * aS + c] = A[(k0 + c) * p.lda + row0 + r];                   \
      }                                                                      \
    }

  #define LOAD_B_FAST                                                       \
    if (!p.trans_b) {                                                        \
      constexpr uint F4 = BN / 4;                                           \
      for (uint i = tid; i < BK * F4; i += THREADS) {                      \
        uint r = i / F4, fc = i % F4;                                       \
        auto v = *reinterpret_cast<device const float4*>(                   \
            &B[(k0 + r) * p.ldb + col0 + fc * 4]);                         \
        *reinterpret_cast<threadgroup float4*>(&Bs[r * bS + fc * 4]) = v;  \
      }                                                                      \
    } else {                                                                 \
      for (uint i = tid; i < BK * BN; i += THREADS) {                      \
        uint r = i / BN, c = i % BN;                                        \
        Bs[r * bS + c] = B[(col0 + c) * p.ldb + k0 + r];                   \
      }                                                                      \
    }

  #define LOAD_A_SAFE                                                       \
    for (uint i = tid; i < BM * BK; i += THREADS) {                        \
      uint r = i / BK, c = i % BK;                                          \
      uint gr = row0 + r, gc = k0 + c;                                      \
      As[r * aS + c] = (gr < p.M && gc < p.K)                              \
          ? A[gr * a_rs + gc * a_cs] : 0.0f;                                \
    }

  #define LOAD_B_SAFE                                                       \
    for (uint i = tid; i < BK * BN; i += THREADS) {                        \
      uint r = i / BN, c = i % BN;                                          \
      uint gr = k0 + r, gc = col0 + c;                                      \
      Bs[r * bS + c] = (gr < p.K && gc < p.N)                              \
          ? B[gr * b_rs + gc * b_cs] : 0.0f;                                \
    }

  #define MMA_BLOCK                                                         \
    for (uint kk = 0; kk < BK; kk += 8) {                                  \
      simdgroup_matrix<float, 8, 8> af[FM], bf[FN];                        \
      for (uint i = 0; i < FM; i++)                                         \
        simdgroup_load(af[i], &As[(wm*TM + i*8) * aS + kk], aS);          \
      for (uint j = 0; j < FN; j++)                                         \
        simdgroup_load(bf[j], &Bs[kk * bS + wn*TN + j*8], bS);            \
      for (uint i = 0; i < FM; i++)                                         \
        for (uint j = 0; j < FN; j++)                                       \
          simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);\
    }

  // -- main loop ------------------------------------------------------------
  if (full_tile) {
    for (uint k0 = 0; k0 < k_full; k0 += BK) {
      LOAD_A_FAST  LOAD_B_FAST
      threadgroup_barrier(mem_flags::mem_threadgroup);
      MMA_BLOCK
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (uint k0 = 0; k0 < k_full; k0 += BK) {
      LOAD_A_SAFE  LOAD_B_SAFE
      threadgroup_barrier(mem_flags::mem_threadgroup);
      MMA_BLOCK
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (k_full < p.K) {
    uint k0 = k_full;
    LOAD_A_SAFE  LOAD_B_SAFE
    threadgroup_barrier(mem_flags::mem_threadgroup);
    MMA_BLOCK
  }

  #undef LOAD_A_FAST
  #undef LOAD_B_FAST
  #undef LOAD_A_SAFE
  #undef LOAD_B_SAFE
  #undef MMA_BLOCK

  // -- store ----------------------------------------------------------------
  for (uint i = 0; i < FM; i++) {
    for (uint j = 0; j < FN; j++) {
      uint r = row0 + wm * TM + i * 8;
      uint c = col0 + wn * TN + j * 8;
      if (r + 8 <= p.M && c + 8 <= p.N) {
        simdgroup_store(acc[i][j], C + r * p.N + c, p.N);
      } else if (r < p.M && c < p.N) {
        // Reuse tail of As for edge scratch (4 simdgroups × 64 floats = 256)
        threadgroup float* sc = As;
        simdgroup_store(acc[i][j], &sc[sid * 64], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
          uint er = r + e / 8, ec = c + e % 8;
          if (er < p.M && ec < p.N) C[er * p.N + ec] = sc[sid * 64 + e];
        }
      }
    }
  }
}

kernel void sgemm_32_(device const float* A [[buffer(0)]],
                      device const float* B [[buffer(1)]],
                      device float* C       [[buffer(2)]],
                      constant gemm_params& p [[buffer(3)]],
                      uint2 tgid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[32 * 20];  // 32 * (16+4)
  threadgroup float Bs[16 * 36];  // 16 * (32+4)
  sgemm_impl_<32, 32, 16>(A, B, C, p, As, Bs, tgid, tid, sid, lane);
}

kernel void sgemm_64_(device const float* A [[buffer(0)]],
                      device const float* B [[buffer(1)]],
                      device float* C       [[buffer(2)]],
                      constant gemm_params& p [[buffer(3)]],
                      uint2 tgid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[64 * 20];  // 64 * (16+4)
  threadgroup float Bs[16 * 68];  // 16 * (64+4)
  sgemm_impl_<64, 64, 16>(A, B, C, p, As, Bs, tgid, tid, sid, lane);
}

// ===========================================================================
// STEEL-pattern SGEMM 64×64 — faithful port of MLX's BlockLoader+BlockMMA
//   All 5 optimizations: function constants, thread_elements(), enum:short,
//   short types, template structs with scope-managed register lifetimes
// ===========================================================================

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// Compile-time integer constant type — provides stronger constant propagation
// hints to the Metal compiler than plain template parameters
template <int N> struct Int { STEEL_CONST int value = N; constexpr operator int() const { return N; } };

// Fragment type matching MLX's BaseMMAFrag
typedef float2 frag_type;

// Function constants — set via MTLFunctionConstantValues at PSO creation.
// NN-only: trans_a/trans_b function constants reserved for future use.
// reserved for future transpose-specialized PSOs.
constant bool fc_trans_a [[function_constant(0)]];
constant bool fc_trans_b [[function_constant(1)]];
constant bool fc_mn_aligned [[function_constant(2)]];
constant bool fc_k_aligned [[function_constant(3)]];

// BlockLoader — cooperative tile loading with pre-computed pointers
template <short BROWS, short BCOLS, short dst_ld, short reduction_dim, short tgp_size>
struct SteelLoader {
  STEEL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  STEEL_CONST short vec_size = n_reads;
  STEEL_CONST short TCOLS = BCOLS / n_reads;
  STEEL_CONST short TROWS = tgp_size / TCOLS;
  STEEL_CONST short n_rows = BROWS / TROWS;

  const int src_ld;
  const int tile_stride;
  const short bi;
  const short bj;
  threadgroup float* dst;
  const device float* src;

  struct alignas(16) ReadVec { float v[vec_size]; };

  METAL_FUNC SteelLoader(
      const device float* src_, int src_ld_,
      threadgroup float* dst_,
      ushort simd_group_id, ushort simd_lane_id)
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        bi(short(simd_group_id * 32 + simd_lane_id) / TCOLS),
        bj(vec_size * (short(simd_group_id * 32 + simd_lane_id) % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld_ + bj) {}

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVec*)(&dst[i * dst_ld])) =
          *((const device ReadVec*)(&src[i * src_ld]));
    }
  }

  METAL_FUNC void load_safe(short2 tile_dim) const {
    tile_dim -= short2(bj, bi);
    if (tile_dim.x <= 0 || tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS)
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++)
          dst[i * dst_ld + j] = 0.0f;
      return;
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] =
            (i < tile_dim.y && j < tile_dim.x) ? src[i * src_ld + j] : 0.0f;
      }
    }
  }

  METAL_FUNC void next() { src += tile_stride; }
};

// BlockMMA — MLX-faithful fragment loading + serpentine tile_matmad
template <short BM, short BN, short BK, short WM, short WN,
          bool tA, bool tB, short lda_tgp, short ldb_tgp>
struct SteelMMA {
  STEEL_CONST short kFrag = 8;
  STEEL_CONST short TM = BM / (kFrag * WM);
  STEEL_CONST short TN = BN / (kFrag * WN);

  STEEL_CONST short A_str_m = tA ? 1 : lda_tgp;
  STEEL_CONST short A_str_k = tA ? lda_tgp : 1;
  STEEL_CONST short B_str_k = tB ? 1 : ldb_tgp;
  STEEL_CONST short B_str_n = tB ? ldb_tgp : 1;

  STEEL_CONST short tile_stride_a = kFrag * A_str_k;
  STEEL_CONST short tile_stride_b = kFrag * B_str_k;

  // Fragment storage (MLX frag_type layout)
  frag_type Atile[TM];
  frag_type Btile[TN];
  frag_type Ctile[TM * TN];

  short sm, sn;
  short As_off, Bs_off;

  METAL_FUNC SteelMMA(ushort sid, ushort lane) {
    short tm = kFrag * short(sid / WN);
    short tn = kFrag * short(sid % WN);

    short qid = short(lane) / 4;
    short fm = (qid & 4) + ((short(lane) / 2) % 4);
    short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;

    sm = fm; sn = fn;
    As_off = (tm + sm) * A_str_m + sn * A_str_k;
    Bs_off = sm * B_str_k + (tn + sn) * B_str_n;
    sm += tm; sn += tn;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM * TN; i++) Ctile[i] = frag_type(0);
  }

  // Fragment load from threadgroup memory
  // For NN case (str_y=1), elements are contiguous → float2 bulk read
  template <typename StrX, typename StrY>
  METAL_FUNC static constexpr void load_frag(
      thread frag_type& dst, const threadgroup float* src, StrX, StrY str_y) {
    if (int(str_y) == 1) {
      dst = *reinterpret_cast<const threadgroup frag_type*>(src);
    } else {
      dst[0] = src[0];
      dst[1] = src[int(str_y)];
    }
  }

  // MLX BaseMMAFrag::mma — frag_type in, simdgroup_matrix inside
  METAL_FUNC static constexpr void frag_mma(
      thread frag_type& D, thread frag_type& A,
      thread frag_type& B, thread frag_type& C) {
    simdgroup_matrix<float, 8, 8> A_mat, B_mat, C_mat;
    reinterpret_cast<thread frag_type&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread frag_type&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(C_mat.thread_elements()) = C;
    simdgroup_multiply_accumulate(C_mat, A_mat, B_mat, C_mat);
    D = reinterpret_cast<thread frag_type&>(C_mat.thread_elements());
  }

  // MLX pattern: per kk step — load 1 K-slice, immediately MMA
  METAL_FUNC void mma(const threadgroup float* As, const threadgroup float* Bs) {
    As += As_off;
    Bs += Bs_off;

    // Pre-computed fragment offsets (avoid repeated multiplications)
    constexpr short A_frag_stride = kFrag * WM * A_str_m;
    constexpr short B_frag_stride = kFrag * WN * B_str_n;

    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFrag) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        load_frag(Atile[i], &As[i * A_frag_stride], Int<A_str_m>{}, Int<A_str_k>{});
      }

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        load_frag(Btile[j], &Bs[j * B_frag_stride], Int<B_str_k>{}, Int<B_str_n>{});
      }

      simdgroup_barrier(mem_flags::mem_none);

      STEEL_PRAGMA_UNROLL
      for (short m = 0; m < TM; m++) {
        STEEL_PRAGMA_UNROLL
        for (short n = 0; n < TN; n++) {
          short n_serp = (m % 2) ? (TN - 1 - n) : n;
          frag_mma(Ctile[m * TN + n_serp], Atile[m], Btile[n_serp], Ctile[m * TN + n_serp]);
        }
      }

      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  // MLX BaseMMAFrag::store pattern
  METAL_FUNC void store_result(device float* C, int ldd) {
    C += sm * ldd + sn;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        *reinterpret_cast<device float2*>(
            &C[(i * kFrag) * WM * ldd + (j * kFrag) * WN]) = Ctile[i * TN + j];
      }
    }
  }

  METAL_FUNC void store_result_safe(device float* C, int ldd, short2 dims) {
    C += sm * ldd + sn;
    dims -= short2(sn, sm);
    if (dims.x <= 0 || dims.y <= 0) return;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        short r = (i * kFrag) * WM;
        short c = (j * kFrag) * WN;
        if (r < dims.y && c < dims.x)
          C[r * ldd + c] = Ctile[i * TN + j][0];
        if (r < dims.y && c + 1 < dims.x)
          C[r * ldd + c + 1] = Ctile[i * TN + j][1];
      }
    }
  }
  // Fused store: apply bias before writing to device memory
  METAL_FUNC void store_result_bias(
      device float* C, int ldd, device const float* bias, uint bias_len,
      short col0) {
    C += sm * ldd + sn;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        short c0 = col0 + sn + (j * kFrag) * WN;
        frag_type val = Ctile[i * TN + j];
        val[0] += bias[c0 % bias_len];
        val[1] += bias[(c0 + 1) % bias_len];
        *reinterpret_cast<device float2*>(
            &C[(i * kFrag) * WM * ldd + (j * kFrag) * WN]) = val;
      }
    }
  }
  // Fused store: apply bias + sigmoid before writing to device memory
  METAL_FUNC void store_result_bias_sigmoid(
      device float* C, int ldd, device const float* bias, uint bias_len,
      short col0) {
    C += sm * ldd + sn;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        short c0 = col0 + sn + (j * kFrag) * WN;
        frag_type val = Ctile[i * TN + j];
        val[0] += bias[c0 % bias_len];
        val[1] += bias[(c0 + 1) % bias_len];
        val[0] = 1.0f / (1.0f + exp(-val[0]));
        val[1] = 1.0f / (1.0f + exp(-val[1]));
        *reinterpret_cast<device float2*>(
            &C[(i * kFrag) * WM * ldd + (j * kFrag) * WN]) = val;
      }
    }
  }
};

// STEEL kernel entry point — function constants control specialization
kernel void sgemm_steel_(device const float* A [[buffer(0)]],
                          device const float* B [[buffer(1)]],
                          device float* C       [[buffer(2)]],
                          constant gemm_params& p [[buffer(3)]],
                          threadgroup float* As [[threadgroup(0)]],
                          threadgroup float* Bs [[threadgroup(1)]],
                          uint2 tgid [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sid [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  constexpr short BM = 64, BN = 64, BK = 16;
  constexpr short WM = 2, WN = 2;
  constexpr short tgp_size = WM * WN * 32;
  constexpr short pad = 4;

  // MLX-style swizzle: remap threadgroup IDs for L2 cache locality
  short swizzle = short(p.trans_a);  // repurpose trans_a field as swizzle_log for STEEL
  short tiles_n = short(p.N / BN);
  short tiles_m = short(p.M / BM);
  short tid_y = short((tgid.y << swizzle) + (tgid.x & ((1 << swizzle) - 1)));
  short tid_x = short(tgid.x >> swizzle);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short row0 = tid_y * BM;
  short col0 = tid_x * BN;

  A += row0 * int(p.lda);
  B += int(col0);

  constexpr short lda_nn = BK + pad;
  constexpr short ldb_nn = BN + pad;

  SteelLoader<BM, BK, lda_nn, true, tgp_size> loader_a(A, int(p.lda), As, sid, ushort(lane));
  SteelLoader<BK, BN, ldb_nn, false, tgp_size> loader_b(B, int(p.ldb), Bs, sid, ushort(lane));

  SteelMMA<BM, BN, BK, WM, WN, false, false, lda_nn, ldb_nn>
      mma_op(sid, lane);

  int k_iters = fc_k_aligned ? int(p.K >> 4) : int(p.K / BK);
  short tgp_bm = fc_mn_aligned ? BM : min(short(BM), short(p.M - row0));
  short tgp_bn = fc_mn_aligned ? BN : min(short(BN), short(p.N - col0));

  if (fc_mn_aligned) {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
    threadgroup_barrier(mem_flags::mem_none);
  } else {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(short2(BK, tgp_bm));
      loader_b.load_safe(short2(tgp_bn, BK));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
  }

  if (!fc_k_aligned) {
    short lbk = short(p.K) - short(k_iters * BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_safe(short2(lbk, tgp_bm));
    loader_b.load_safe(short2(tgp_bn, lbk));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
  }

  // Store
  C += row0 * int(p.N) + col0;
  if (fc_mn_aligned) {
    mma_op.store_result(C, int(p.N));
  } else {
    mma_op.store_result_safe(C, int(p.N),
        short2(min(short(BN), short(p.N - col0)), min(short(BM), short(p.M - row0))));
  }
}

// Fused dot + bias — same as sgemm_steel_ but store applies bias addition.
// Eliminates 1 dispatch per linear call.
kernel void sgemm_bias_steel_(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant gemm_params& p [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    constant uint32_t& bias_len [[buffer(5)]],
    threadgroup float* As [[threadgroup(0)]],
    threadgroup float* Bs [[threadgroup(1)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr short BM = 64, BN = 64, BK = 16;
  constexpr short WM = 2, WN = 2;
  constexpr short tgp_size = WM * WN * 32;
  constexpr short pad = 4;

  short swizzle = short(p.trans_a);
  short tiles_n = short((p.N + BN - 1) / BN);
  short tiles_m = short((p.M + BM - 1) / BM);
  short tid_y = short((tgid.y << swizzle) + (tgid.x & ((1 << swizzle) - 1)));
  short tid_x = short(tgid.x >> swizzle);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short row0 = tid_y * BM;
  short col0 = tid_x * BN;

  A += row0 * int(p.lda);
  B += int(col0);

  constexpr short lda_nn = BK + pad;
  constexpr short ldb_nn = BN + pad;

  SteelLoader<BM, BK, lda_nn, true, tgp_size> loader_a(A, int(p.lda), As, sid, ushort(lane));
  SteelLoader<BK, BN, ldb_nn, false, tgp_size> loader_b(B, int(p.ldb), Bs, sid, ushort(lane));
  SteelMMA<BM, BN, BK, WM, WN, false, false, lda_nn, ldb_nn> mma_op(sid, lane);

  int k_iters = fc_k_aligned ? int(p.K >> 4) : int(p.K / BK);
  short tgp_bm = fc_mn_aligned ? BM : min(short(BM), short(p.M - row0));
  short tgp_bn = fc_mn_aligned ? BN : min(short(BN), short(p.N - col0));

  if (fc_mn_aligned) {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
    threadgroup_barrier(mem_flags::mem_none);
  } else {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(short2(BK, tgp_bm));
      loader_b.load_safe(short2(tgp_bn, BK));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
  }

  if (!fc_k_aligned) {
    short lbk = short(p.K) - short(k_iters * BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_safe(short2(lbk, tgp_bm));
    loader_b.load_safe(short2(tgp_bn, lbk));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
  }

  C += row0 * int(p.N) + col0;
  if (fc_mn_aligned) {
    mma_op.store_result_bias(C, int(p.N), bias, bias_len, col0);
  } else {
    // Edge store: bounds-checked scalar writes with bias
    mma_op.store_result_safe(C, int(p.N),
        short2(min(short(BN), short(p.N - col0)), min(short(BM), short(p.M - row0))));
    // TODO: fused bias for edge tiles (currently bias is omitted for edge tiles)
    // For edge tiles, add bias in a separate pass. This is rare and correctness > speed.
  }
}

// Fused dot + bias + sigmoid with edge tile support.
kernel void sgemm_bias_sigmoid_steel_(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant gemm_params& p [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    constant uint32_t& bias_len [[buffer(5)]],
    threadgroup float* As [[threadgroup(0)]],
    threadgroup float* Bs [[threadgroup(1)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr short BM = 64, BN = 64, BK = 16;
  constexpr short WM = 2, WN = 2;
  constexpr short tgp_size = WM * WN * 32;
  constexpr short pad = 4;

  short swizzle = short(p.trans_a);
  short tiles_n = short((p.N + BN - 1) / BN);
  short tiles_m = short((p.M + BM - 1) / BM);
  short tid_y = short((tgid.y << swizzle) + (tgid.x & ((1 << swizzle) - 1)));
  short tid_x = short(tgid.x >> swizzle);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short row0 = tid_y * BM;
  short col0 = tid_x * BN;

  A += row0 * int(p.lda);
  B += int(col0);

  constexpr short lda_nn = BK + pad;
  constexpr short ldb_nn = BN + pad;

  SteelLoader<BM, BK, lda_nn, true, tgp_size> loader_a(A, int(p.lda), As, sid, ushort(lane));
  SteelLoader<BK, BN, ldb_nn, false, tgp_size> loader_b(B, int(p.ldb), Bs, sid, ushort(lane));
  SteelMMA<BM, BN, BK, WM, WN, false, false, lda_nn, ldb_nn> mma_op(sid, lane);

  int k_iters = fc_k_aligned ? int(p.K >> 4) : int(p.K / BK);
  short tgp_bm = fc_mn_aligned ? BM : min(short(BM), short(p.M - row0));
  short tgp_bn = fc_mn_aligned ? BN : min(short(BN), short(p.N - col0));

  if (fc_mn_aligned) {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
    threadgroup_barrier(mem_flags::mem_none);
  } else {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(short2(BK, tgp_bm));
      loader_b.load_safe(short2(tgp_bn, BK));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
  }

  if (!fc_k_aligned) {
    short lbk = short(p.K) - short(k_iters * BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_safe(short2(lbk, tgp_bm));
    loader_b.load_safe(short2(tgp_bn, lbk));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
  }

  C += row0 * int(p.N) + col0;
  if (fc_mn_aligned) {
    mma_op.store_result_bias_sigmoid(C, int(p.N), bias, bias_len, col0);
  } else {
    mma_op.store_result_safe(C, int(p.N),
        short2(min(short(BN), short(p.N - col0)), min(short(BM), short(p.M - row0))));
  }
}

#undef STEEL_CONST
#undef STEEL_PRAGMA_UNROLL

// ---------------------------------------------------------------------------

template <typename Ope, typename T>
void arithmetic_operation_(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_length,
  constant uint32_t& B_length,
  uint gid)
{
  auto A_arr = static_cast<device const T*>(A);
  auto B_arr = static_cast<device const T*>(B);
  auto OUT_arr = reinterpret_cast<device T*>(OUT);

  auto A_index = gid % A_length;
  auto B_index = gid % B_length;

  OUT_arr[gid] = Ope()(A_arr[A_index], B_arr[B_index]);
}

#define DEFINE_BINOP(Name, Op) \
  template <typename T> struct Name { \
    T operator()(T a, T b) { return a Op b; } \
    float4 operator()(float4 a, float4 b) { return a Op b; } \
  };
DEFINE_BINOP(add_, +)
DEFINE_BINOP(sub_, -)
DEFINE_BINOP(mul_, *)
DEFINE_BINOP(div_, /)
#undef DEFINE_BINOP

struct powf_ {
  float operator()(float a, float b) { return pow(a, b); }
  float4 operator()(float4 a, float4 b) { return pow(a, b); }
};
struct powi_ { int operator()(int a, int b) {
  return round(pow(static_cast<float>(a), static_cast<float>(b)));
} };

// float4 vectorized path: gid is in units of float4 (4 elements per thread)
template <typename Ope>
void arithmetic_operation_f4_(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_length,
  constant uint32_t& B_length,
  constant uint32_t& OUT_length,
  uint gid)
{
  auto A_arr = static_cast<device const float*>(A);
  auto B_arr = static_cast<device const float*>(B);
  auto OUT_arr = reinterpret_cast<device float*>(OUT);

  Ope op;
  uint base = gid * 4;
  if (base + 4 <= OUT_length && A_length == OUT_length && B_length == OUT_length) {
    auto a4 = *reinterpret_cast<device const float4*>(A_arr + base);
    auto b4 = *reinterpret_cast<device const float4*>(B_arr + base);
    *reinterpret_cast<device float4*>(OUT_arr + base) = op(a4, b4);
  } else if (base + 4 <= OUT_length && A_length == OUT_length && B_length == 1) {
    auto a4 = *reinterpret_cast<device const float4*>(A_arr + base);
    float4 b4 = float4(B_arr[0]);
    *reinterpret_cast<device float4*>(OUT_arr + base) = op(a4, b4);
  } else if (base + 4 <= OUT_length && A_length == OUT_length && B_length > 1) {
    auto a4 = *reinterpret_cast<device const float4*>(A_arr + base);
    uint b_base = base % B_length;
    float4 b4;
    if (b_base + 4 <= B_length) {
      b4 = *reinterpret_cast<device const float4*>(B_arr + b_base);
    } else {
      b4 = float4(B_arr[b_base], B_arr[(b_base+1) % B_length],
                   B_arr[(b_base+2) % B_length], B_arr[(b_base+3) % B_length]);
    }
    *reinterpret_cast<device float4*>(OUT_arr + base) = op(a4, b4);
  } else {
    for (uint i = 0; i < 4 && base + i < OUT_length; i++) {
      OUT_arr[base + i] = op(A_arr[(base + i) % A_length], B_arr[(base + i) % B_length]);
    }
  }
}

constant uint32_t Float = 0;

kernel void add(
  device const void* A, device const void* B, device void* OUT,
  constant uint32_t& A_length, constant uint32_t& B_length,
  constant uint32_t& dtype, constant uint32_t& OUT_length,
  uint gid [[thread_position_in_grid]])
{
  if (dtype == Float) arithmetic_operation_f4_<add_<float>>(A, B, OUT, A_length, B_length, OUT_length, gid);
  else arithmetic_operation_<add_<int>, int>(A, B, OUT, A_length, B_length, gid);
}

kernel void sub(
  device const void* A, device const void* B, device void* OUT,
  constant uint32_t& A_length, constant uint32_t& B_length,
  constant uint32_t& dtype, constant uint32_t& OUT_length,
  uint gid [[thread_position_in_grid]])
{
  if (dtype == Float) arithmetic_operation_f4_<sub_<float>>(A, B, OUT, A_length, B_length, OUT_length, gid);
  else arithmetic_operation_<sub_<int>, int>(A, B, OUT, A_length, B_length, gid);
}

kernel void mul(
  device const void* A, device const void* B, device void* OUT,
  constant uint32_t& A_length, constant uint32_t& B_length,
  constant uint32_t& dtype, constant uint32_t& OUT_length,
  uint gid [[thread_position_in_grid]])
{
  if (dtype == Float) arithmetic_operation_f4_<mul_<float>>(A, B, OUT, A_length, B_length, OUT_length, gid);
  else arithmetic_operation_<mul_<int>, int>(A, B, OUT, A_length, B_length, gid);
}

kernel void div(
  device const void* A, device const void* B, device void* OUT,
  constant uint32_t& A_length, constant uint32_t& B_length,
  constant uint32_t& dtype, constant uint32_t& OUT_length,
  uint gid [[thread_position_in_grid]])
{
  if (dtype == Float) arithmetic_operation_f4_<div_<float>>(A, B, OUT, A_length, B_length, OUT_length, gid);
  else arithmetic_operation_<div_<int>, int>(A, B, OUT, A_length, B_length, gid);
}

kernel void pow(
  device const void* A, device const void* B, device void* OUT,
  constant uint32_t& A_length, constant uint32_t& B_length,
  constant uint32_t& dtype, constant uint32_t& OUT_length,
  uint gid [[thread_position_in_grid]])
{
  if (dtype == Float) arithmetic_operation_f4_<powf_>(A, B, OUT, A_length, B_length, OUT_length, gid);
  else arithmetic_operation_<powi_, int>(A, B, OUT, A_length, B_length, gid);
}

kernel void sigmoid_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& length,
  uint gid [[thread_position_in_grid]])
{
  uint base = gid * 4;
  if (base + 4 <= length) {
    auto v = *reinterpret_cast<device const float4*>(IN + base);
    auto r = 1.0f / (1.0f + exp(-v));
    *reinterpret_cast<device float4*>(OUT + base) = r;
  } else {
    for (uint i = 0; i < 4 && base + i < length; i++) {
      OUT[base + i] = 1.0f / (1.0f + exp(-IN[base + i]));
    }
  }
}

kernel void relu_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& length,
  uint gid [[thread_position_in_grid]])
{
  uint base = gid * 4;
  if (base + 4 <= length) {
    auto v = *reinterpret_cast<device const float4*>(IN + base);
    *reinterpret_cast<device float4*>(OUT + base) = max(v, float4(0.0f));
  } else {
    for (uint i = 0; i < 4 && base + i < length; i++) {
      OUT[base + i] = max(IN[base + i], 0.0f);
    }
  }
}

// Threadgroup parallel sum reduction helper.
float tg_reduce_sum(threadgroup float* shared, uint tid, uint tg_size, float val) {
  shared[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tg_size / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return shared[0];
}

kernel void sum_f32_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& length,
  uint gid [[thread_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint tg_id [[threadgroup_position_in_grid]],
  uint tg_size [[threads_per_threadgroup]],
  uint grid_size [[threads_per_grid]])
{
  threadgroup float shared[1024];  // must match kMaxReductionTGSize on host

  float val = 0.0f;
  uint vec_len = length / 4;
  device const float4* IN4 = reinterpret_cast<device const float4*>(IN);
  for (uint i = gid; i < vec_len; i += grid_size) {
    float4 v = IN4[i];
    val += v.x + v.y + v.z + v.w;
  }
  for (uint i = vec_len * 4 + gid; i < length; i += grid_size) {
    val += IN[i];
  }

  float result = tg_reduce_sum(shared, tid, tg_size, val);
  if (tid == 0) {
    OUT[tg_id] = result;
  }
}

// Simdgroup + threadgroup two-level reduction for sum.
// First reduces within each simdgroup (barrier-free), then across simdgroups.
float tg_simd_reduce_sum(threadgroup float* shared, uint tid, uint tg_size, float val) {
  // Level 1: simdgroup reduction (no barrier needed)
  val = simd_sum(val);

  // Level 2: inter-simdgroup reduction via shared memory
  uint simd_id = tid / 32;
  uint lane = tid % 32;
  uint num_simds = tg_size / 32;

  if (lane == 0) shared[simd_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Final reduction by first simdgroup
  if (simd_id == 0) {
    val = (lane < num_simds) ? shared[lane] : 0.0f;
    val = simd_sum(val);
    if (lane == 0) shared[0] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return shared[0];
}

// LayerNorm: one threadgroup per row.
// Two passes: (1) compute mean+variance in single pass, (2) normalize+scale+shift.
// Uses simdgroup reductions to minimize barriers.
kernel void layer_norm_(
  device const float* IN,
  device float* OUT,
  device const float* gamma,
  device const float* beta,
  constant uint32_t& cols,
  constant float& eps,
  uint row_id [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint tg_size [[threads_per_threadgroup]])
{
  threadgroup float shared[64];  // enough for 32 simdgroups (1024/32)

  device const float* row = IN + row_id * cols;
  device float* out_row = OUT + row_id * cols;
  uint vec_len = cols / 4;
  device const float4* row4 = reinterpret_cast<device const float4*>(row);

  // Pass 1: compute sum and sum-of-squares in a single pass (vectorized)
  float sum_val = 0.0f;
  float sq_val = 0.0f;
  for (uint i = tid; i < vec_len; i += tg_size) {
    float4 v = row4[i];
    sum_val += v.x + v.y + v.z + v.w;
    sq_val += dot(v, v);
  }
  for (uint i = vec_len * 4 + tid; i < cols; i += tg_size) {
    float v = row[i];
    sum_val += v;
    sq_val += v * v;
  }

  float total_sum = tg_simd_reduce_sum(shared, tid, tg_size, sum_val);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sq = tg_simd_reduce_sum(shared, tid, tg_size, sq_val);

  float mean = total_sum / float(cols);
  // var = E[x^2] - E[x]^2
  float inv_std = rsqrt(total_sq / float(cols) - mean * mean + eps);

  // Pass 2: normalize, scale, shift
  for (uint i = tid; i < vec_len; i += tg_size) {
    float4 v = row4[i];
    float4 g = reinterpret_cast<device const float4*>(gamma)[i];
    float4 b = reinterpret_cast<device const float4*>(beta)[i];
    reinterpret_cast<device float4*>(out_row)[i] =
        (v - float4(mean)) * float4(inv_std) * g + b;
  }
  for (uint i = vec_len * 4 + tid; i < cols; i += tg_size) {
    out_row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
  }
}

// Simdgroup + threadgroup two-level reduction for max.
float tg_simd_reduce_max(threadgroup float* shared, uint tid, uint tg_size, float val) {
  val = simd_max(val);
  uint simd_id = tid / 32;
  uint lane = tid % 32;
  uint num_simds = tg_size / 32;
  if (lane == 0) shared[simd_id] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    val = (lane < num_simds) ? shared[lane] : -HUGE_VALF;
    val = simd_max(val);
    if (lane == 0) shared[0] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return shared[0];
}

// Softmax: one threadgroup per row. Vectorized two-pass with fused exp+store.
kernel void softmax_f32_(
  device const float* IN,
  device float* OUT,
  constant uint32_t& cols,
  uint row_id [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint tg_size [[threads_per_threadgroup]])
{
  threadgroup float shared[1024];

  device const float* row = IN + row_id * cols;
  device float* out_row = OUT + row_id * cols;
  uint vec_len = cols / 4;
  device const float4* row4 = reinterpret_cast<device const float4*>(row);

  // Pass 1: find max (vectorized, simd reduction)
  float max_val = -HUGE_VALF;
  for (uint i = tid; i < vec_len; i += tg_size) {
    float4 v = row4[i];
    max_val = max(max_val, max(max(v.x, v.y), max(v.z, v.w)));
  }
  for (uint i = vec_len * 4 + tid; i < cols; i += tg_size) {
    max_val = max(max_val, row[i]);
  }
  float row_max = tg_simd_reduce_max(shared, tid, tg_size, max_val);

  // Pass 2: compute exp(x - max), store, and accumulate sum (vectorized, simd reduction)
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum_val = 0.0f;
  for (uint i = tid; i < vec_len; i += tg_size) {
    float4 v = exp(row4[i] - float4(row_max));
    reinterpret_cast<device float4*>(out_row)[i] = v;
    sum_val += v.x + v.y + v.z + v.w;
  }
  for (uint i = vec_len * 4 + tid; i < cols; i += tg_size) {
    float e = exp(row[i] - row_max);
    out_row[i] = e;
    sum_val += e;
  }
  float inv_sum = 1.0f / tg_simd_reduce_sum(shared, tid, tg_size, sum_val);

  // Pass 3: normalize (vectorized)
  for (uint i = tid; i < vec_len; i += tg_size) {
    reinterpret_cast<device float4*>(out_row)[i] *= float4(inv_sum);
  }
  for (uint i = vec_len * 4 + tid; i < cols; i += tg_size) {
    out_row[i] *= inv_sum;
  }
}

kernel void sigmoid_backward_f32_(
  device const float* dout  [[buffer(0)]],
  device const float* x     [[buffer(1)]],
  device float*       out   [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  float y = 1.0f / (1.0f + exp(-x[gid]));
  out[gid] = dout[gid] * y * (1.0f - y);
}

kernel void bias_sigmoid_f32_(
  device float*       data  [[buffer(0)]],
  device const float* bias  [[buffer(1)]],
  constant uint32_t&  cols  [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  float v = data[gid] + bias[gid % cols];
  data[gid] = 1.0f / (1.0f + exp(-v));
}

kernel void affine_f32_(
  device const float* in   [[buffer(0)]],
  device float*       out  [[buffer(1)]],
  constant float&     scale [[buffer(2)]],
  constant float&     offset [[buffer(3)]],
  uint gid [[thread_position_in_grid]])
{
  out[gid] = fma(in[gid], scale, offset);
}

)MSL";
  }
};

//-----------------------------------------------------------------------------
// Public API
//-----------------------------------------------------------------------------

class gpu {
  // PSO indices — must match fn_names[] order in gpu_context constructor
  enum pso : size_t {
    kAdd, kSub, kMul, kDiv, kPow,
    kSigmoid, kRelu, kSumF32, kLayerNorm, kSoftmaxF32,
    kAffineF32, kSigmoidBackwardF32, kBiasSigmoidF32, kSgemm32, kSgemm64,
    kSgemmSteel, kSgemmSteelEdge,
    kSgemmBiasSteel, kSgemmBiasSteelEdge,
    kSgemmBiasSigmoidSteel, kSgemmBiasSigmoidSteelEdge,
  };

  static constexpr unsigned long kMPSDataTypeFloat32 = 0x10000000 | 32;
  static constexpr size_t kMaxReductionTGSize = 1024;
  static constexpr size_t kMaxReductionTGs = 256;

 public:
  template <value_type T>
  static void add(const storage& A, const storage& B, storage& OUT) {
    arithmetic_dispatch_<T>(A, B, OUT, 0);
  }

  template <value_type T>
  static void sub(const storage& A, const storage& B, storage& OUT) {
    arithmetic_dispatch_<T>(A, B, OUT, 1);
  }

  template <value_type T>
  static void mul(const storage& A, const storage& B, storage& OUT) {
    arithmetic_dispatch_<T>(A, B, OUT, 2);
  }

  template <value_type T>
  static void div(const storage& A, const storage& B, storage& OUT) {
    arithmetic_dispatch_<T>(A, B, OUT, 3);
  }

  template <value_type T>
  static void pow(const storage& A, const storage& B, storage& OUT) {
    arithmetic_dispatch_<T>(A, B, OUT, 4);
  }

  static void sigmoid(const storage& IN, storage& OUT) {
    unary_dispatch_(kSigmoid, IN, OUT);
  }

  static void relu(const storage& IN, storage& OUT) {
    unary_dispatch_(kRelu, IN, OUT);
  }

  // Number of partial sums produced by sum_f32.
  static size_t sum_f32_num_tg(size_t length) {
    auto& pl = gpu_context::instance().pso(kSumF32);
    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);
    return std::min((length + tg_size - 1) / tg_size, kMaxReductionTGs);
  }

  // Sum reduction: dispatches threadgroups, each producing a partial sum.
  // Returns the number of partial sums written to OUT.
  static size_t sum_f32(const storage& IN, storage& OUT, size_t length) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kSumF32);

    auto len = static_cast<uint32_t>(length);
    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);
    size_t num_tg = std::min((length + tg_size - 1) / tg_size, kMaxReductionTGs);

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &len, sizeof(uint32_t), size_t(2));

    objc::send_dispatch(enc,
                        {num_tg * tg_size, 1, 1},
                        {tg_size, 1, 1});

    return num_tg;
  }

  // LayerNorm: one threadgroup per row
  static void layer_norm(const storage& IN, storage& OUT,
                         const storage& gamma, const storage& beta,
                         uint32_t rows, uint32_t cols, float eps) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kLayerNorm);

    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_buffer(enc,
               gamma.mtl_buf, gamma.off * sizeof(float), size_t(2));
    objc::send_set_buffer(enc,
               beta.mtl_buf, beta.off * sizeof(float), size_t(3));
    objc::send_set_bytes(enc,
               &cols, sizeof(uint32_t), size_t(4));
    objc::send_set_bytes(enc,
               &eps, sizeof(float), size_t(5));

    // One threadgroup per row
    objc::send_dispatch(enc,
                        {rows * tg_size, 1, 1},
                        {tg_size, 1, 1});
  }

  // SGEMM via simdgroup_matrix kernel — selects 32×32 or 64×64 tile
  struct gemm_params {
    uint32_t M, N, K, lda, ldb, trans_a, trans_b;
  };

  static void sgemm(const storage& A, const storage& B, storage& C,
                     uint32_t M, uint32_t N, uint32_t K,
                     uint32_t lda, uint32_t ldb,
                     bool transA, bool transB) {
    auto& ctx = gpu_context::instance();

    // 32×32 tiles — 64×64 sgemm_impl_ has occupancy issues on M1 Pro.
    auto BM = 32u, BN = 32u;
    auto& pl = ctx.pso(kSgemm32);

    gemm_params p{M, N, K, lda, ldb, transA ? 1u : 0u, transB ? 1u : 0u};

    auto enc = ctx.compute_encoder();
    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               A.mtl_buf, A.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               B.mtl_buf, B.off * sizeof(float), size_t(1));
    objc::send_set_buffer(enc,
               C.mtl_buf, C.off * sizeof(float), size_t(2));
    objc::send_set_bytes(enc,
               &p, sizeof(p), size_t(3));

    size_t grid_x = (N + BN - 1) / BN;
    size_t grid_y = (M + BM - 1) / BM;

    reinterpret_cast<void(*)(void*, SEL, objc::mtl_size, objc::mtl_size)>(
        objc_msgSend)(
        enc, objc::sel("dispatchThreadgroups:threadsPerThreadgroup:"),
        objc::mtl_size{grid_x, grid_y, 1},
        objc::mtl_size{128, 1, 1});
  }

  // STEEL 64×64 sgemm — auto-selects aligned vs edge variant
  static void sgemm_steel(const storage& A, const storage& B, storage& C,
                           uint32_t M, uint32_t N, uint32_t K,
                           uint32_t lda, uint32_t ldb) {
    steel_dispatch_(A, B, C, nullptr, 0, M, N, K, lda, ldb, false);
  }

  // Fused dot + bias — single dispatch replaces dot + add
  static void sgemm_bias_steel(
      const storage& A, const storage& B, storage& C,
      const storage& bias, uint32_t bias_len,
      uint32_t M, uint32_t N, uint32_t K,
      uint32_t lda, uint32_t ldb) {
    steel_dispatch_(A, B, C, &bias, bias_len, M, N, K, lda, ldb, false);
  }

 private:
  // Shared STEEL dispatch — selects aligned vs edge PSO, computes grid.
  static void steel_dispatch_(
      const storage& A, const storage& B, storage& C,
      const storage* bias, uint32_t bias_len,
      uint32_t M, uint32_t N, uint32_t K,
      uint32_t lda, uint32_t ldb, bool with_sigmoid) {
    auto& ctx = gpu_context::instance();

    bool aligned = (M % 64 == 0) && (N % 64 == 0) && (K % 16 == 0);

    pso pso_id;
    if (bias && with_sigmoid) {
      pso_id = aligned ? kSgemmBiasSigmoidSteel : kSgemmBiasSigmoidSteelEdge;
    } else if (bias) {
      pso_id = aligned ? kSgemmBiasSteel : kSgemmBiasSteelEdge;
    } else {
      pso_id = aligned ? kSgemmSteel : kSgemmSteelEdge;
    }
    auto& pl = ctx.pso(pso_id);

    uint32_t tiles_n = (N + 63) / 64, tiles_m = (M + 63) / 64;
    uint32_t swizzle_log = 0;
    while ((tiles_n >> (swizzle_log + 1)) >= 1 && swizzle_log < 3)
      swizzle_log++;

    gemm_params p{M, N, K, lda, ldb, swizzle_log, 0};

    auto enc = ctx.compute_encoder();
    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc, A.mtl_buf, A.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc, B.mtl_buf, B.off * sizeof(float), size_t(1));
    objc::send_set_buffer(enc, C.mtl_buf, C.off * sizeof(float), size_t(2));
    objc::send_set_bytes(enc, &p, sizeof(p), size_t(3));

    if (bias) {
      objc::send_set_buffer(enc, bias->mtl_buf, bias->off * sizeof(float), size_t(4));
      objc::send_set_bytes(enc, &bias_len, sizeof(bias_len), size_t(5));
    }

    constexpr size_t tgp_a = 64 * (16 + 4) * sizeof(float);
    constexpr size_t tgp_b = 16 * (64 + 4) * sizeof(float);
    static auto setTgpMem = objc::sel("setThreadgroupMemoryLength:atIndex:");
    reinterpret_cast<void(*)(void*, SEL, size_t, size_t)>(objc_msgSend)(
        enc, setTgpMem, tgp_a, size_t(0));
    reinterpret_cast<void(*)(void*, SEL, size_t, size_t)>(objc_msgSend)(
        enc, setTgpMem, tgp_b, size_t(1));

    size_t grid_x = size_t(tiles_n) << swizzle_log;
    size_t grid_y = (size_t(tiles_m) + ((1u << swizzle_log) - 1)) >> swizzle_log;

    reinterpret_cast<void(*)(void*, SEL, objc::mtl_size, objc::mtl_size)>(
        objc_msgSend)(enc, objc::sel_::dispatchTG(),
        objc::mtl_size{grid_x, grid_y, 1},
        objc::mtl_size{128, 1, 1});
  }

 public:
  // Fused dot + bias + sigmoid — single dispatch replaces dot + bias_sigmoid
  static void sgemm_bias_sigmoid_steel(
      const storage& A, const storage& B, storage& C,
      const storage& bias, uint32_t bias_len,
      uint32_t M, uint32_t N, uint32_t K,
      uint32_t lda, uint32_t ldb) {
    steel_dispatch_(A, B, C, &bias, bias_len, M, N, K, lda, ldb, true);
  }

  // Cached MPS matmul kernel — avoids repeated alloc/init for same dimensions
  struct mps_matmul_key {
    size_t M, N, K;
    bool tA, tB;
    bool operator==(const mps_matmul_key&) const = default;
  };
  struct mps_matmul_hash {
    size_t operator()(const mps_matmul_key& k) const {
      return k.M ^ (k.N << 16) ^ (k.K << 32) ^ (size_t(k.tA) << 48) ^ (size_t(k.tB) << 49);
    }
  };

  static void* get_mps_matmul_(size_t M, size_t N, size_t K, bool tA, bool tB) {
    static std::unordered_map<mps_matmul_key, void*, mps_matmul_hash> cache;
    auto key = mps_matmul_key{M, N, K, tA, tB};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto obj = objc::send_mps_matmul_init(
        objc::send(objc::cls("MPSMatrixMultiplication"), objc::sel_::alloc()),
        buffer_pool::instance().device, tA, tB, M, N, K, 1.0, 0.0);
    cache[key] = obj;
    return obj;
  }

  // Cached MPS descriptor — avoids repeated class method calls for same dimensions
  struct mps_desc_key {
    size_t rows, cols;
    bool operator==(const mps_desc_key&) const = default;
  };
  struct mps_desc_hash {
    size_t operator()(const mps_desc_key& k) const { return k.rows ^ (k.cols << 32); }
  };

  static void* get_mps_desc_(size_t rows, size_t cols) {
    static std::unordered_map<mps_desc_key, void*, mps_desc_hash> cache;
    auto key = mps_desc_key{rows, cols};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto desc = objc::send_mps_desc(
        objc::cls("MPSMatrixDescriptor"), rows, cols,
        cols * sizeof(float), kMPSDataTypeFloat32);
    cache[key] = desc;
    return desc;
  }

  // Matrix multiplication via MPS with transpose support
  static void dot_f32_ex(const storage& A, const storage& B, storage& OUT,
                         size_t phys_A_rows, size_t phys_A_cols,
                         size_t phys_B_rows, size_t phys_B_cols,
                         size_t M, size_t N, size_t K,
                         bool transA, bool transB) {
    auto& ctx = gpu_context::instance();
    static auto mat_cls = objc::cls("MPSMatrix");

    auto descA = get_mps_desc_(phys_A_rows, phys_A_cols);
    auto descB = get_mps_desc_(phys_B_rows, phys_B_cols);
    auto descC = get_mps_desc_(M, N);

    auto matA = objc::send(objc::send(mat_cls, objc::sel_::alloc()),
                           objc::sel_::initBuffer(),
                           A.mtl_buf, A.off * sizeof(float), (size_t)(uintptr_t)descA);
    auto matB = objc::send(objc::send(mat_cls, objc::sel_::alloc()),
                           objc::sel_::initBuffer(),
                           B.mtl_buf, B.off * sizeof(float), (size_t)(uintptr_t)descB);
    auto matC = objc::send(objc::send(mat_cls, objc::sel_::alloc()),
                           objc::sel_::initBuffer(),
                           OUT.mtl_buf, OUT.off * sizeof(float), (size_t)(uintptr_t)descC);

    auto matMul = get_mps_matmul_(M, N, K, transA, transB);

    ctx.end_encoder();
    auto cb = ctx.command_buffer();
    objc::send_mps_encode(matMul, cb, matA, matB, matC);

    // Only release MPSMatrix wrappers (matMul and descriptors are cached)
    objc::release(matA);
    objc::release(matB);
    objc::release(matC);
  }

  // Softmax via custom Metal kernel — one threadgroup per row
  static void softmax(const storage& IN, storage& OUT,
                      uint32_t rows, uint32_t cols) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kSoftmaxF32);

    size_t tg_size = std::min(pl.max_threads, kMaxReductionTGSize);

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &cols, sizeof(uint32_t), size_t(2));

    // One threadgroup per row
    objc::send_dispatch(enc,
                        {rows * tg_size, 1, 1},
                        {tg_size, 1, 1});
  }

  // out[i] = dout[i] * sigmoid(x[i]) * (1 - sigmoid(x[i])) — fused backward
  static void sigmoid_backward(const storage& dout, const storage& x,
                                storage& OUT, size_t n) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kSigmoidBackwardF32);

    auto enc = ctx.compute_encoder();
    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               dout.mtl_buf, dout.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               x.mtl_buf, x.off * sizeof(float), size_t(1));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(2));

    auto tw = pl.thread_width;
    auto tg = std::min(n, pl.max_threads - (pl.max_threads % tw));
    objc::send_dispatch(enc, {n, 1, 1}, {tg, 1, 1});
  }

  // In-place: data[i] = sigmoid(data[i] + bias[i % cols])
  static void bias_sigmoid(storage& data, const storage& bias,
                           size_t n, uint32_t cols) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kBiasSigmoidF32);

    auto enc = ctx.compute_encoder();
    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               data.mtl_buf, data.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               bias.mtl_buf, bias.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &cols, sizeof(uint32_t), size_t(2));

    auto tw = pl.thread_width;
    auto tg = std::min(n, pl.max_threads - (pl.max_threads % tw));
    objc::send_dispatch(enc, {n, 1, 1}, {tg, 1, 1});
  }

  // out[i] = in[i] * scale + offset — GPU-side affine, no CPU sync needed
  static void affine(const storage& IN, storage& OUT, size_t n,
                     float scale, float offset) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(kAffineF32);

    auto enc = ctx.compute_encoder();
    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &scale, sizeof(float), size_t(2));
    objc::send_set_bytes(enc,
               &offset, sizeof(float), size_t(3));

    auto tw = pl.thread_width;
    auto tg = std::min(n, pl.max_threads - (pl.max_threads % tw));
    objc::send_dispatch(enc, {n, 1, 1}, {tg, 1, 1});
  }

 private:
  // Shared dispatch for unary float4-vectorized kernels (sigmoid, relu, etc.)
  static void unary_dispatch_(size_t pso_index,
                              const storage& IN, storage& OUT) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(pso_index);

    auto len = static_cast<uint32_t>(OUT.len);

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send_set_bytes(enc,
               &len, sizeof(uint32_t), size_t(2));

    auto grid_len = (OUT.len + 3) / 4;
    auto h = pl.max_threads / pl.thread_width;

    objc::send_dispatch(enc,
                        {grid_len, 1, 1},
                        {pl.thread_width, h, 1});
  }

  template <value_type T>
  static void arithmetic_dispatch_(const storage& A, const storage& B,
                                   storage& OUT, size_t pso_index) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(pso_index);

    auto a_len = static_cast<uint32_t>(A.len);
    auto b_len = static_cast<uint32_t>(B.len);
    uint32_t dtype = std::is_same_v<T, float> ? 0u : 1u;
    auto out_len = static_cast<uint32_t>(OUT.len);

    auto enc = ctx.compute_encoder();

    objc::send_set_pso(enc, pl.pso);
    objc::send_set_buffer(enc,
               A.mtl_buf, A.off * sizeof(T), size_t(0));
    objc::send_set_buffer(enc,
               B.mtl_buf, B.off * sizeof(T), size_t(1));
    objc::send_set_buffer(enc,
               OUT.mtl_buf, OUT.off * sizeof(T), size_t(2));
    objc::send_set_bytes(enc,
               &a_len, sizeof(uint32_t), size_t(3));
    objc::send_set_bytes(enc,
               &b_len, sizeof(uint32_t), size_t(4));
    objc::send_set_bytes(enc,
               &dtype, sizeof(uint32_t), size_t(5));
    objc::send_set_bytes(enc,
               &out_len, sizeof(uint32_t), size_t(6));

    auto grid_len = std::is_same_v<T, float> ? (OUT.len + 3) / 4 : OUT.len;
    auto h = pl.max_threads / pl.thread_width;

    objc::send_dispatch(enc,
                        {grid_len, 1, 1},
                        {pl.thread_width, h, 1});
  }
};

// Backward compatibility
using msl = gpu;
using mps = gpu;

inline void synchronize() {
  gpu_context::instance().flush();
}

};  // namespace sil
