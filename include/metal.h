#pragma once

#include <common.h>

#include <Metal/Metal.hpp>

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

};  // namespace mtlcpp::mtl

