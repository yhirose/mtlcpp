#include <metal.h>

#include "doctest.h"
#include "utils.h"

using namespace mtl;

template <typename T, typename U>
bool verify(const mtl::managed_ptr<MTL::Buffer> &A,
            const mtl::managed_ptr<MTL::Buffer> &B,
            const mtl::managed_ptr<MTL::Buffer> &OUT, U fn) {
  return verify(static_cast<T *>(A->contents()),
                static_cast<T *>(B->contents()),
                static_cast<T *>(OUT->contents()), A->length() / sizeof(T), fn);
}

template <typename T, typename U>
bool verify_tolerant(const mtl::managed_ptr<MTL::Buffer> &A,
                     const mtl::managed_ptr<MTL::Buffer> &B,
                     const mtl::managed_ptr<MTL::Buffer> &OUT, U fn) {
  return verify_tolerant(
      static_cast<T *>(A->contents()), static_cast<T *>(B->contents()),
      static_cast<T *>(OUT->contents()), A->length() / sizeof(T), fn);
}

template <typename T>
void random(mtl::managed_ptr<MTL::Buffer> &buf) {
  auto p = static_cast<T *>(buf->contents());
  auto arr_len = buf->length() / sizeof(T);
  for (size_t i = 0; i < arr_len; i++) {
    p[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
  }
}

TEST_CASE("testing basic operations") {
  auto device = mtl::managed(MTL::CreateSystemDefaultDevice());
  auto metal = mtl::metal(device.get());

  size_t arr_len = 60 * 180 * 10000;
  size_t buf_len = arr_len * sizeof(float);

  auto A = metal.newBuffer(buf_len);
  auto B = metal.newBuffer(buf_len);
  auto OUT = metal.newBuffer(buf_len);

  random<float>(A);
  random<float>(B);

  metal.compute(A.get(), B.get(), OUT.get(), mtl::ComputeType::ARRAY_ADD_F,
                sizeof(float));
  CHECK(verify<float>(A, B, OUT, [](auto a, auto b) { return a + b; }));

  metal.compute(A.get(), B.get(), OUT.get(), mtl::ComputeType::ARRAY_SUB_F,
                sizeof(float));
  CHECK(verify<float>(A, B, OUT, [](auto a, auto b) { return a - b; }));

  metal.compute(A.get(), B.get(), OUT.get(), mtl::ComputeType::ARRAY_MUL_F,
                sizeof(float));
  CHECK(verify<float>(A, B, OUT, [](auto a, auto b) { return a * b; }));

  metal.compute(A.get(), B.get(), OUT.get(), mtl::ComputeType::ARRAY_DIV_F,
                sizeof(float));
  CHECK(
      verify_tolerant<float>(A, B, OUT, [](auto a, auto b) { return a / b; }));
}
