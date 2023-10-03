#include "gpu.h"

#include "metal.h"

namespace mtlcpp {

mtl::metal &mltcpp() {
  static auto device_ = mtl::managed(MTL::CreateSystemDefaultDevice());
  static auto mltcpp_ = mtl::metal(device_.get());
  return mltcpp_;
}

class GPU::MemoryImpl {
 public:
  void *data() const { return buf_->contents(); }
  size_t length() const { return buf_->length(); }

 private:
  friend class GPU;
  mtl::managed_ptr<MTL::Buffer> buf_;
};

GPU::Memory::Memory() : impl_(new MemoryImpl) {}

void *GPU::Memory::data() { return impl_->data(); }

void const *GPU::Memory::data() const { return impl_->data(); }

size_t GPU::Memory::length() const { return impl_->length(); }

GPU::Memory GPU::allocate(size_t buf_len) {
  Memory buf;
  buf.impl_->buf_ = mltcpp().newBuffer(buf_len);
  return buf;
}

void GPU::compute(const GPU::Memory &A, const GPU::Memory &B, const GPU::Memory &OUT,
                  ComputeType id, size_t element_size) {
  mltcpp().compute(A.impl_->buf_.get(), B.impl_->buf_.get(),
                   OUT.impl_->buf_.get(), id, element_size);
}

}  // namespace mtlcpp
