#pragma once

#include <common.h>

#include <Metal/Metal.hpp>
#include <sstream>

namespace mtl {

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

//-----------------------------------------------------------------------------

class metal {
 public:
  metal(MTL::Device* device);

  managed_ptr<MTL::Buffer> newBuffer(NS::UInteger length);

  template <value_type T>
  void compute(MTL::Buffer* A, size_t A_offset, uint32_t A_bytes,
               MTL::Buffer* B, size_t B_offset, uint32_t B_bytes,
               MTL::Buffer* OUT, size_t OUT_offset, uint32_t OUT_bytes,
               Operation ope);

 private:
  MTL::Device* device_;
  std::vector<managed_ptr<MTL::ComputePipelineState>> states_;
  managed_ptr<MTL::CommandQueue> queue_;

  void create_compute_pipeline_state_object_(MTL::Device* device,
                                             managed_ptr<MTL::Library> library,
                                             const char* name);
};

//-----------------------------------------------------------------------------

static const char* metal_source_ = R"(

#include <metal_stdlib>

using namespace metal;

template <typename T>
uint broadcast_offset(uint index, uint bytes)
{
  return index % (bytes / sizeof(T));
}

template <typename Ope, typename T>
void compute(
  device const T* A,
  device const T* B,
  device T* OUT,
  constant uint32_t& A_bytes,
  constant uint32_t& B_bytes,
  uint index)
{
  OUT[index] = Ope()(
    A[broadcast_offset<T>(index, A_bytes)],
    B[broadcast_offset<T>(index, B_bytes)]);
}

template <typename T> struct add { T operator()(T a, T b) { return a + b; } };
template <typename T> struct sub { T operator()(T a, T b) { return a - b; } };
template <typename T> struct mul { T operator()(T a, T b) { return a * b; } };
template <typename T> struct div { T operator()(T a, T b) { return a / b; } };

constant uint32_t Float = 0;

kernel void array_add(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_bytes,
  constant uint32_t& B_bytes,
  constant uint32_t& dtype,
  uint index [[thread_position_in_grid]])
{
  if (dtype == Float) {
    compute<add<float>>(
      static_cast<device const float*>(A),
      static_cast<device const float*>(B),
      static_cast<device float*>(OUT),
      A_bytes, B_bytes, index);
  } else {
    compute<add<int>>(
      static_cast<device const int*>(A),
      static_cast<device const int*>(B),
      static_cast<device int*>(OUT),
      A_bytes, B_bytes, index);
  }
}

kernel void array_sub(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_bytes,
  constant uint32_t& B_bytes,
  constant uint32_t& dtype,
  uint index [[thread_position_in_grid]])
{
  if (dtype == Float) {
    compute<sub<float>>(
      static_cast<device const float*>(A),
      static_cast<device const float*>(B),
      static_cast<device float*>(OUT),
      A_bytes, B_bytes, index);
  } else {
    compute<sub<int>>(
      static_cast<device const int*>(A),
      static_cast<device const int*>(B),
      static_cast<device int*>(OUT),
      A_bytes, B_bytes, index);
  }
}

kernel void array_mul(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_bytes,
  constant uint32_t& B_bytes,
  constant uint32_t& dtype,
  uint index [[thread_position_in_grid]])
{
  if (dtype == Float) {
    compute<mul<float>>(
      static_cast<device const float*>(A),
      static_cast<device const float*>(B),
      static_cast<device float*>(OUT),
      A_bytes, B_bytes, index);
  } else {
    compute<mul<int>>(
      static_cast<device const int*>(A),
      static_cast<device const int*>(B),
      static_cast<device int*>(OUT),
      A_bytes, B_bytes, index);
  }
}

kernel void array_div(
  device const void* A,
  device const void* B,
  device void* OUT,
  constant uint32_t& A_bytes,
  constant uint32_t& B_bytes,
  constant uint32_t& dtype,
  uint index [[thread_position_in_grid]])
{
  if (dtype == Float) {
    compute<div<float>>(
      static_cast<device const float*>(A),
      static_cast<device const float*>(B),
      static_cast<device float*>(OUT),
      A_bytes, B_bytes, index);
  } else {
    compute<div<int>>(
      static_cast<device const int*>(A),
      static_cast<device const int*>(B),
      static_cast<device int*>(OUT),
      A_bytes, B_bytes, index);
  }
}

)";

//-----------------------------------------------------------------------------

inline metal::metal(MTL::Device* device) : device_(device) {
  if (device == nullptr) {
    throw std::runtime_error("metal: Failed to create the default library.");
    return;
  }

  // Compile a Metal library
  auto src = NS::String::string(metal_source_, NS::ASCIIStringEncoding);
  NS::Error* error = nullptr;

  auto lib = managed(device->newLibrary(src, nullptr, &error));
  if (lib == nullptr) {
    std::stringstream ss;
    ss << "metal: Failed to compile the Metal library, error " << error << ".";
    throw std::runtime_error(ss.str());
    return;
  }

  // Create pipeline state objects
  create_compute_pipeline_state_object_(device, lib, "array_add");
  create_compute_pipeline_state_object_(device, lib, "array_sub");
  create_compute_pipeline_state_object_(device, lib, "array_mul");
  create_compute_pipeline_state_object_(device, lib, "array_div");

  // Create a command queue
  queue_ = managed(device->newCommandQueue());

  if (queue_ == nullptr) {
    throw std::runtime_error("metal: Failed to find the command queue.");
    return;
  }
}

inline managed_ptr<MTL::Buffer> metal::newBuffer(NS::UInteger length) {
  return managed(device_->newBuffer(length, MTL::ResourceStorageModeShared));
}

template <value_type T>
inline void metal::compute(MTL::Buffer* A, size_t A_offset, uint32_t A_bytes,
                           MTL::Buffer* B, size_t B_offset, uint32_t B_bytes,
                           MTL::Buffer* OUT, size_t OUT_offset,
                           uint32_t OUT_bytes, Operation ope) {
  auto commandBuffer = queue_->commandBuffer();
  auto computeEncoder = commandBuffer->computeCommandEncoder();

  auto pso = states_[static_cast<size_t>(ope)];
  auto dtype = std::is_same_v<T, float>
                   ? static_cast<uint32_t>(DataType::Float)
                   : static_cast<uint32_t>(DataType::Integer);

  computeEncoder->setComputePipelineState(pso.get());
  computeEncoder->setBuffer(A, A_offset, 0);
  computeEncoder->setBuffer(B, B_offset, 1);
  computeEncoder->setBuffer(OUT, OUT_offset, 2);
  computeEncoder->setBytes(&A_bytes, sizeof(uint32_t), 3);
  computeEncoder->setBytes(&B_bytes, sizeof(uint32_t), 4);
  computeEncoder->setBytes(&dtype, sizeof(uint32_t), 5);

  auto length = OUT_bytes / sizeof(T);
  auto grid_size = MTL::Size::Make(length, 1, 1);
  auto threads_size = std::min(pso->maxTotalThreadsPerThreadgroup(), length);

  computeEncoder->dispatchThreads(grid_size,
                                  MTL::Size::Make(threads_size, 1, 1));

  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

inline void metal::create_compute_pipeline_state_object_(
    MTL::Device* device, managed_ptr<MTL::Library> library, const char* name) {
  auto str = NS::String::string(name, NS::ASCIIStringEncoding);
  auto fn = managed(library->newFunction(str));

  if (fn == nullptr) {
    std::stringstream ss;
    ss << "metal: Failed to find the " << name << " function.";
    throw std::runtime_error(ss.str());
    return;
  }

  NS::Error* error = nullptr;
  auto pso = managed(device->newComputePipelineState(fn.get(), &error));

  if (pso == nullptr) {
    std::stringstream ss;
    ss << "metal: Failed to created pipeline state object, error " << error
       << ".";
    throw std::runtime_error(ss.str());
    return;
  }

  states_.push_back(pso);
}

inline metal& singleton_instance_() {
  static auto device_ = managed(MTL::CreateSystemDefaultDevice());
  static auto metal_ = metal(device_.get());
  return metal_;
}

inline managed_ptr<MTL::Buffer> newBuffer(NS::UInteger length) {
  return singleton_instance_().newBuffer(length);
}

template <typename T>
inline void compute(managed_ptr<MTL::Buffer> A, size_t A_offset,
                    uint32_t A_bytes, managed_ptr<MTL::Buffer> B,
                    size_t B_offset, uint32_t B_bytes,
                    managed_ptr<MTL::Buffer> OUT, size_t OUT_offset,
                    uint32_t OUT_bytes, Operation ope) {
  return singleton_instance_().compute<T>(
      A.get(), A_offset, A->length(), B.get(), B_offset, B->length(), OUT.get(),
      OUT_offset, OUT->length(), ope);
}

};  // namespace mtl
