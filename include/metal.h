#pragma once

#include <Metal/Metal.hpp>
#include <sstream>

namespace mtlcpp::mtl {

//-----------------------------------------------------------------------------

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

enum class ComputeType {
  ARRAY_ADD_F = 0,
  ARRAY_SUB_F,
  ARRAY_MUL_F,
  ARRAY_DIV_F,
  ARRAY_ADD_I,
  ARRAY_SUB_I,
  ARRAY_MUL_I,
  ARRAY_DIV_I,
  ARRAY_ADD_U,
  ARRAY_SUB_U,
  ARRAY_MUL_U,
  ARRAY_DIV_U,
};

class metal {
 public:
  metal(MTL::Device* device);

  managed_ptr<MTL::Buffer> newBuffer(NS::UInteger length);

  void compute(MTL::Buffer* A, MTL::Buffer* B, MTL::Buffer* OUT, ComputeType id,
               size_t element_size);

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

kernel void array_add_f(
  device const float* A,
  device const float* B,
  device float* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] + B[index];
}

kernel void array_sub_f(
  device const float* A,
  device const float* B,
  device float* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] - B[index];
}

kernel void array_mul_f(
  device const float* A,
  device const float* B,
  device float* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] * B[index];
}

kernel void array_div_f(
  device const float* A,
  device const float* B,
  device float* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] / B[index];
}

kernel void array_add_i(
  device const int* A,
  device const int* B,
  device int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] + B[index];
}

kernel void array_sub_i(
  device const int* A,
  device const int* B,
  device int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] - B[index];
}

kernel void array_mul_i(
  device const int* A,
  device const int* B,
  device int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] * B[index];
}

kernel void array_div_i(
  device const int* A,
  device const int* B,
  device int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] / B[index];
}

kernel void array_add_u(
  device const unsigned int* A,
  device const unsigned int* B,
  device unsigned int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] + B[index];
}

kernel void array_sub_u(
  device const unsigned int* A,
  device const unsigned int* B,
  device unsigned int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] - B[index];
}

kernel void array_mul_u(
  device const unsigned int* A,
  device const unsigned int* B,
  device unsigned int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] * B[index];
}

kernel void array_div_u(
  device const unsigned int* A,
  device const unsigned int* B,
  device unsigned int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] / B[index];
}

)";

//-----------------------------------------------------------------------------

inline metal::metal(MTL::Device* device) : device_(device) {
  if (device == nullptr) {
    std::stringstream ss;
    ss << "metal: Failed to create the default library.";
    throw std::runtime_error(ss.str());
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
  create_compute_pipeline_state_object_(device, lib, "array_add_f");
  create_compute_pipeline_state_object_(device, lib, "array_sub_f");
  create_compute_pipeline_state_object_(device, lib, "array_mul_f");
  create_compute_pipeline_state_object_(device, lib, "array_div_f");
  create_compute_pipeline_state_object_(device, lib, "array_add_i");
  create_compute_pipeline_state_object_(device, lib, "array_sub_i");
  create_compute_pipeline_state_object_(device, lib, "array_mul_i");
  create_compute_pipeline_state_object_(device, lib, "array_div_i");
  create_compute_pipeline_state_object_(device, lib, "array_add_u");
  create_compute_pipeline_state_object_(device, lib, "array_sub_u");
  create_compute_pipeline_state_object_(device, lib, "array_mul_u");
  create_compute_pipeline_state_object_(device, lib, "array_div_u");

  // Create a command queue
  queue_ = managed(device->newCommandQueue());

  if (queue_ == nullptr) {
    std::stringstream ss;
    ss << "metal: Failed to find the command queue.";
    throw std::runtime_error(ss.str());
    return;
  }
}

inline managed_ptr<MTL::Buffer> metal::newBuffer(NS::UInteger length) {
  return managed(device_->newBuffer(length, MTL::ResourceStorageModeShared));
}

inline void metal::compute(MTL::Buffer* A, MTL::Buffer* B, MTL::Buffer* OUT,
                           ComputeType id, size_t element_size) {
  auto commandBuffer = queue_->commandBuffer();

  auto computeEncoder = commandBuffer->computeCommandEncoder();

  auto pso = states_[static_cast<size_t>(id)];

  computeEncoder->setComputePipelineState(pso.get());
  computeEncoder->setBuffer(A, 0, 0);
  computeEncoder->setBuffer(B, 0, 1);
  computeEncoder->setBuffer(OUT, 0, 2);

  auto length = A->length() / element_size;
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

inline void compute(MTL::Buffer* A, MTL::Buffer* B, MTL::Buffer* OUT,
                    ComputeType id, size_t element_size) {
  return singleton_instance_().compute(A, B, OUT, id, element_size);
}

};  // namespace mtlcpp::mtl

