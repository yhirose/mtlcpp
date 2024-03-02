#pragma once

#include <Metal/Metal.hpp>
#include <numeric>
#include <sstream>

namespace mtl {

template <typename T>
concept value_type =
    std::same_as<T, float> || std::same_as<T, int> || std::same_as<T, bool>;

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template <typename T>
struct releaser {
  void operator()(T* p) {
    if (p != nullptr) {
      p->release();
    } else {
      throw std::runtime_error(
          "metal: This managed resource object has already been released...");
    }
  }
};

template <typename T>
inline auto managed(T* p) {
  return std::shared_ptr<T>(p, releaser<T>());
}

template <typename T>
using managed_ptr = std::shared_ptr<T>;

struct storage {
  managed_ptr<MTL::Buffer> buf;
  size_t off = 0;
  size_t len = 0;
};

//-----------------------------------------------------------------------------

class metal {
 public:
  metal(MTL::Device* device);

  managed_ptr<MTL::Buffer> make_buffer(NS::UInteger length);

  template <value_type T>
  void add(const storage& A, const storage& B, storage& OUT);

  template <value_type T>
  void sub(const storage& A, const storage& B, storage& OUT);

  template <value_type T>
  void mul(const storage& A, const storage& B, storage& OUT);

  template <value_type T>
  void div(const storage& A, const storage& B, storage& OUT);

  template <value_type T>
  void dot(const storage& A, const storage& B, storage& OUT, uint32_t OUT_rows,
           uint32_t OUT_cols, uint32_t m);

 private:
  enum class DataType {
    Float = 0,
    Integer,
  };

  MTL::Device* device_;
  managed_ptr<MTL::CommandQueue> queue_;

  managed_ptr<MTL::ComputePipelineState> pso_add_;
  managed_ptr<MTL::ComputePipelineState> pso_sub_;
  managed_ptr<MTL::ComputePipelineState> pso_mul_;
  managed_ptr<MTL::ComputePipelineState> pso_div_;
  managed_ptr<MTL::ComputePipelineState> pso_dot_;

  template <value_type T>
  void arithmetic_operation_(const storage& A, const storage& B, storage& OUT,
                             managed_ptr<MTL::ComputePipelineState> pso);

  managed_ptr<MTL::ComputePipelineState> create_compute_pipeline_state_object_(
      MTL::Device* device, managed_ptr<MTL::Library> library, const char* name);
};

//-----------------------------------------------------------------------------

static const char* metal_source_ = R"(

#include <metal_stdlib>

using namespace metal;

template <typename Ope, typename T>
void arithmetic_operation_(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_length,
  constant uint32_t& B_length,
  uint index)
{
  auto A_arr = static_cast<device const T*>(A);
  auto B_arr = static_cast<device const T*>(B);
  auto OUT_arr = reinterpret_cast<device T*>(OUT);

  // broadcast offset
  auto A_index = index % A_length;
  auto B_index = index % B_length;

  OUT_arr[index] = Ope()(A_arr[A_index], B_arr[B_index]);
}

template <typename T> struct add_ { T operator()(T a, T b) { return a + b; } };
template <typename T> struct sub_ { T operator()(T a, T b) { return a - b; } };
template <typename T> struct mul_ { T operator()(T a, T b) { return a * b; } };
template <typename T> struct div_ { T operator()(T a, T b) { return a / b; } };

template <typename T>
void dot_operatoin(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& OUT_rows,
  constant uint32_t& OUT_cols,
  constant uint32_t& m,
  uint index)
{
  auto A_arr = static_cast<device const T*>(A);
  auto B_arr = static_cast<device const T*>(B);
  auto OUT_arr = reinterpret_cast<device T*>(OUT);

  auto irow = index / OUT_cols;
  auto icol = index % OUT_cols;

  T val{};
  for (uint32_t i = 0; i < m; i++) {
    auto aval = A_arr[m * irow + i];
    auto bval = B_arr[OUT_cols * i + icol];
    val += aval * bval;
  }

  OUT_arr[index] = val;
}

constant uint32_t Float = 0;

kernel void add(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_length,
  constant uint32_t& B_length,
  constant uint32_t& dtype,
  uint index [[thread_position_in_grid]])
{
  if (dtype == Float) {
    arithmetic_operation_<add_<float>, float>(A, B, OUT, A_length, B_length, index);
  } else {
    arithmetic_operation_<add_<int>, int>(A, B, OUT, A_length, B_length, index);
  }
}

kernel void sub(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_length,
  constant uint32_t& B_length,
  constant uint32_t& dtype,
  uint index [[thread_position_in_grid]])
{
  if (dtype == Float) {
    arithmetic_operation_<sub_<float>, float>(A, B, OUT, A_length, B_length, index);
  } else {
    arithmetic_operation_<sub_<int>, int>(A, B, OUT, A_length, B_length, index);
  }
}

kernel void mul(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_length,
  constant uint32_t& B_length,
  constant uint32_t& dtype,
  uint index [[thread_position_in_grid]])
{
  if (dtype == Float) {
    arithmetic_operation_<mul_<float>, float>(A, B, OUT, A_length, B_length, index);
  } else {
    arithmetic_operation_<mul_<int>, int>(A, B, OUT, A_length, B_length, index);
  }
}

kernel void div(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_length,
  constant uint32_t& B_length,
  constant uint32_t& dtype,
  uint index [[thread_position_in_grid]])
{
  if (dtype == Float) {
    arithmetic_operation_<div_<float>, float>(A, B, OUT, A_length, B_length, index);
  } else {
    arithmetic_operation_<div_<int>, int>(A, B, OUT, A_length, B_length, index);
  }
}

kernel void dot(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& OUT_rows,
  constant uint32_t& OUT_cols,
  constant uint32_t& m,
  constant uint32_t& dtype,
  uint index [[thread_position_in_grid]])
{
  if (dtype == Float) {
    dot_operatoin<float>(A, B, OUT, OUT_rows, OUT_cols, m, index);
  } else {
    dot_operatoin<int>(A, B, OUT, OUT_rows, OUT_cols, m, index);
  }
}

)";

//-----------------------------------------------------------------------------

inline metal::metal(MTL::Device* device) : device_(device) {
  if (device == nullptr) {
    throw std::runtime_error("metal: Failed to create the default library.");
  }

  // Compile a Metal library
  auto src = NS::String::string(metal_source_, NS::ASCIIStringEncoding);
  NS::Error* error = nullptr;

  auto lib = managed(device->newLibrary(src, nullptr, &error));
  if (lib == nullptr) {
    std::stringstream ss;
    ss << "metal: Failed to compile the Metal library, error " << error << ".";
    throw std::runtime_error(ss.str());
  }

  // Create pipeline state objects
  pso_add_ = create_compute_pipeline_state_object_(device, lib, "add");
  pso_sub_ = create_compute_pipeline_state_object_(device, lib, "sub");
  pso_mul_ = create_compute_pipeline_state_object_(device, lib, "mul");
  pso_div_ = create_compute_pipeline_state_object_(device, lib, "div");
  pso_dot_ = create_compute_pipeline_state_object_(device, lib, "dot");

  // Create a command queue
  queue_ = managed(device->newCommandQueue());

  if (queue_ == nullptr) {
    throw std::runtime_error("metal: Failed to find the command queue.");
  }
}

inline managed_ptr<MTL::Buffer> metal::make_buffer(NS::UInteger length) {
  return managed(device_->newBuffer(length, MTL::ResourceStorageModeShared));
}

template <value_type T>
inline void metal::add(const storage& A, const storage& B, storage& OUT) {
  arithmetic_operation_<T>(A, B, OUT, pso_add_);
}

template <value_type T>
inline void metal::sub(const storage& A, const storage& B, storage& OUT) {
  arithmetic_operation_<T>(A, B, OUT, pso_sub_);
}

template <value_type T>
inline void metal::mul(const storage& A, const storage& B, storage& OUT) {
  arithmetic_operation_<T>(A, B, OUT, pso_mul_);
}

template <value_type T>
inline void metal::div(const storage& A, const storage& B, storage& OUT) {
  arithmetic_operation_<T>(A, B, OUT, pso_div_);
}

template <value_type T>
inline void metal::dot(const storage& A, const storage& B, storage& OUT,
                       uint32_t OUT_rows, uint32_t OUT_cols, uint32_t m) {
  auto pso = pso_dot_;
  auto dtype = std::is_same_v<T, float>
                   ? static_cast<uint32_t>(DataType::Float)
                   : static_cast<uint32_t>(DataType::Integer);

  auto commandBuffer = queue_->commandBuffer();
  auto computeEncoder = commandBuffer->computeCommandEncoder();

  computeEncoder->setComputePipelineState(pso.get());
  computeEncoder->setBuffer(A.buf.get(), A.off * sizeof(T), 0);
  computeEncoder->setBuffer(B.buf.get(), B.off * sizeof(T), 1);
  computeEncoder->setBuffer(OUT.buf.get(), OUT.off * sizeof(T), 2);
  computeEncoder->setBytes(&OUT_rows, sizeof(uint32_t), 3);
  computeEncoder->setBytes(&OUT_cols, sizeof(uint32_t), 4);
  computeEncoder->setBytes(&m, sizeof(uint32_t), 5);
  computeEncoder->setBytes(&dtype, sizeof(uint32_t), 6);

  auto grid_size = MTL::Size::Make(OUT.len, 1, 1);
  auto threads_size =
      std::min<size_t>(pso->maxTotalThreadsPerThreadgroup(), OUT.len);

  computeEncoder->dispatchThreads(grid_size,
                                  MTL::Size::Make(threads_size, 1, 1));

  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

template <value_type T>
inline void metal::arithmetic_operation_(
    const storage& A, const storage& B, storage& OUT,
    managed_ptr<MTL::ComputePipelineState> pso) {
  auto dtype = std::is_same_v<T, float>
                   ? static_cast<uint32_t>(DataType::Float)
                   : static_cast<uint32_t>(DataType::Integer);

  auto commandBuffer = queue_->commandBuffer();
  auto computeEncoder = commandBuffer->computeCommandEncoder();

  computeEncoder->setComputePipelineState(pso.get());
  computeEncoder->setBuffer(A.buf.get(), A.off * sizeof(T), 0);
  computeEncoder->setBuffer(B.buf.get(), B.off * sizeof(T), 1);
  computeEncoder->setBuffer(OUT.buf.get(), OUT.off * sizeof(T), 2);
  computeEncoder->setBytes(&A.len, sizeof(uint32_t), 3);
  computeEncoder->setBytes(&B.len, sizeof(uint32_t), 4);
  computeEncoder->setBytes(&dtype, sizeof(uint32_t), 5);

  auto grid_size = MTL::Size::Make(OUT.len, 1, 1);
  auto threads_size =
      std::min<size_t>(pso->maxTotalThreadsPerThreadgroup(), OUT.len);

  computeEncoder->dispatchThreads(grid_size,
                                  MTL::Size::Make(threads_size, 1, 1));

  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

inline managed_ptr<MTL::ComputePipelineState>
metal::create_compute_pipeline_state_object_(MTL::Device* device,
                                             managed_ptr<MTL::Library> library,
                                             const char* name) {
  auto str = NS::String::string(name, NS::ASCIIStringEncoding);
  auto fn = managed(library->newFunction(str));

  if (fn == nullptr) {
    std::stringstream ss;
    ss << "metal: Failed to find the " << name << " function.";
    throw std::runtime_error(ss.str());
  }

  NS::Error* error = nullptr;
  auto pso = managed(device->newComputePipelineState(fn.get(), &error));

  if (pso == nullptr) {
    std::stringstream ss;
    ss << "metal: Failed to created pipeline state object, error " << error
       << ".";
    throw std::runtime_error(ss.str());
  }

  return pso;
}

inline metal& singleton_instance_() {
  static auto device_ = managed(MTL::CreateSystemDefaultDevice());
  static auto metal_ = metal(device_.get());
  return metal_;
}

//-----------------------------------------------------------------------------

inline managed_ptr<MTL::Buffer> make_buffer(NS::UInteger length) {
  return singleton_instance_().make_buffer(length);
}

template <typename T>
inline void add(const storage& A, const storage& B, storage& OUT) {
  return singleton_instance_().add<T>(A, B, OUT);
}

template <typename T>
inline void sub(const storage& A, const storage& B, storage& OUT) {
  return singleton_instance_().sub<T>(A, B, OUT);
}

template <typename T>
inline void mul(const storage& A, const storage& B, storage& OUT) {
  return singleton_instance_().mul<T>(A, B, OUT);
}

template <typename T>
inline void div(const storage& A, const storage& B, storage& OUT) {
  return singleton_instance_().div<T>(A, B, OUT);
}

template <typename T>
inline void dot(const storage& A, const storage& B, storage& OUT,
                uint32_t OUT_rows, uint32_t OUT_cols, uint32_t m) {
  return singleton_instance_().dot<T>(A, B, OUT, OUT_rows, OUT_cols, m);
}

};  // namespace mtl
