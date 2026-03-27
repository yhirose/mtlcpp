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

 private:
  std::vector<pipeline> psos_;

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
    const char* fn_names[] = {"add", "sub", "mul", "div", "pow", "sigmoid_"};
    for (auto name : fn_names) {
      psos_.push_back(create_pso_(device, lib, name));
    }

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

  static const char* msl_source_() {
    return R"(

#include <metal_stdlib>

using namespace metal;

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

template <typename T> struct add_ { T operator()(T a, T b) { return a + b; } };
template <typename T> struct sub_ { T operator()(T a, T b) { return a - b; } };
template <typename T> struct mul_ { T operator()(T a, T b) { return a * b; } };
template <typename T> struct div_ { T operator()(T a, T b) { return a / b; } };

struct powf_ { float operator()(float a, float b) { return pow(a, b); } };
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
    *reinterpret_cast<device float4*>(OUT_arr + base) =
        float4(op(a4.x, b4.x), op(a4.y, b4.y), op(a4.z, b4.z), op(a4.w, b4.w));
  } else if (base + 4 <= OUT_length && A_length == OUT_length && B_length == 1) {
    auto a4 = *reinterpret_cast<device const float4*>(A_arr + base);
    float b_val = B_arr[0];
    *reinterpret_cast<device float4*>(OUT_arr + base) =
        float4(op(a4.x, b_val), op(a4.y, b_val), op(a4.z, b_val), op(a4.w, b_val));
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

)";
  }
};

//-----------------------------------------------------------------------------
// Public API
//-----------------------------------------------------------------------------

class gpu {
 public:
  // Elementwise operations
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

  // Sigmoid
  static void sigmoid(const storage& IN, storage& OUT) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(5);  // sigmoid_ is at index 5

    auto len = static_cast<uint32_t>(OUT.len);

    auto cb = objc::send(ctx.queue, "commandBuffer");
    auto enc = objc::send(cb, "computeCommandEncoder");

    objc::send(enc, "setComputePipelineState:", pl.pso);
    objc::send(enc, "setBuffer:offset:atIndex:",
               IN.mtl_buf, IN.off * sizeof(float), size_t(0));
    objc::send(enc, "setBuffer:offset:atIndex:",
               OUT.mtl_buf, OUT.off * sizeof(float), size_t(1));
    objc::send(enc, "setBytes:length:atIndex:",
               &len, sizeof(uint32_t), size_t(2));

    auto grid_len = (OUT.len + 3) / 4;
    auto h = pl.max_threads / pl.thread_width;

    objc::send_dispatch(enc,
                        {grid_len, 1, 1},
                        {pl.thread_width, h, 1});

    objc::send(enc, "endEncoding");
    objc::send(cb, "commit");
    objc::send(cb, "waitUntilCompleted");
  }

  // Matrix multiplication via MPS
  static void dot_f32(const storage& A, const storage& B, storage& OUT,
                      uint32_t A_cols, uint32_t OUT_rows, uint32_t OUT_cols) {
    auto& pool = buffer_pool::instance();
    auto& ctx = gpu_context::instance();

    constexpr unsigned long MPSDataTypeFloat32 = 0x10000000 | 32;

    auto mps_desc_cls = objc::cls("MPSMatrixDescriptor");
    auto mps_mat_cls = objc::cls("MPSMatrix");
    auto mps_matmul_cls = objc::cls("MPSMatrixMultiplication");

    auto descA = objc::send_mps_desc(mps_desc_cls, OUT_rows, A_cols,
                                      A_cols * sizeof(float), MPSDataTypeFloat32);
    auto descB = objc::send_mps_desc(mps_desc_cls, A_cols, OUT_cols,
                                      OUT_cols * sizeof(float), MPSDataTypeFloat32);
    auto descC = objc::send_mps_desc(mps_desc_cls, OUT_rows, OUT_cols,
                                      OUT_cols * sizeof(float), MPSDataTypeFloat32);

    auto matA = objc::send(objc::send(mps_mat_cls, "alloc"),
                           "initWithBuffer:offset:descriptor:",
                           A.mtl_buf, A.off * sizeof(float), (size_t)(uintptr_t)descA);
    auto matB = objc::send(objc::send(mps_mat_cls, "alloc"),
                           "initWithBuffer:offset:descriptor:",
                           B.mtl_buf, B.off * sizeof(float), (size_t)(uintptr_t)descB);
    auto matC = objc::send(objc::send(mps_mat_cls, "alloc"),
                           "initWithBuffer:offset:descriptor:",
                           OUT.mtl_buf, OUT.off * sizeof(float), (size_t)(uintptr_t)descC);

    auto matMul = objc::send_mps_matmul_init(
        objc::send(mps_matmul_cls, "alloc"),
        pool.device, false, false,
        OUT_rows, OUT_cols, A_cols, 1.0, 0.0);

    auto cb = objc::send(ctx.queue, "commandBuffer");
    objc::send_mps_encode(matMul, cb, matA, matB, matC);
    objc::send(cb, "commit");
    objc::send(cb, "waitUntilCompleted");

    objc::release(matMul);
    objc::release(matA);
    objc::release(matB);
    objc::release(matC);
  }

 private:
  template <value_type T>
  static void arithmetic_dispatch_(const storage& A, const storage& B,
                                   storage& OUT, size_t pso_index) {
    auto& ctx = gpu_context::instance();
    auto& pl = ctx.pso(pso_index);

    auto a_len = static_cast<uint32_t>(A.len);
    auto b_len = static_cast<uint32_t>(B.len);
    uint32_t dtype = std::is_same_v<T, float> ? 0u : 1u;
    auto out_len = static_cast<uint32_t>(OUT.len);

    auto cb = objc::send(ctx.queue, "commandBuffer");
    auto enc = objc::send(cb, "computeCommandEncoder");

    objc::send(enc, "setComputePipelineState:", pl.pso);
    objc::send(enc, "setBuffer:offset:atIndex:",
               A.mtl_buf, A.off * sizeof(T), size_t(0));
    objc::send(enc, "setBuffer:offset:atIndex:",
               B.mtl_buf, B.off * sizeof(T), size_t(1));
    objc::send(enc, "setBuffer:offset:atIndex:",
               OUT.mtl_buf, OUT.off * sizeof(T), size_t(2));
    objc::send(enc, "setBytes:length:atIndex:",
               &a_len, sizeof(uint32_t), size_t(3));
    objc::send(enc, "setBytes:length:atIndex:",
               &b_len, sizeof(uint32_t), size_t(4));
    objc::send(enc, "setBytes:length:atIndex:",
               &dtype, sizeof(uint32_t), size_t(5));
    objc::send(enc, "setBytes:length:atIndex:",
               &out_len, sizeof(uint32_t), size_t(6));

    auto grid_len = std::is_same_v<T, float> ? (OUT.len + 3) / 4 : OUT.len;
    auto h = pl.max_threads / pl.thread_width;

    objc::send_dispatch(enc,
                        {grid_len, 1, 1},
                        {pl.thread_width, h, 1});

    objc::send(enc, "endEncoding");
    objc::send(cb, "commit");
    objc::send(cb, "waitUntilCompleted");
  }
};

// Backward compatibility
using msl = gpu;
using mps = gpu;

};  // namespace sil
