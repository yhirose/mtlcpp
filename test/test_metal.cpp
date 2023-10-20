#include <metal.h>

#include "doctest.h"
#include "utils.h"

using namespace mtl;

template <typename T, typename U>
bool verify_array(const managed_ptr<MTL::Buffer> &A,
                  const managed_ptr<MTL::Buffer> &B,
                  const managed_ptr<MTL::Buffer> &OUT, U fn) {
  return verify_array(
      static_cast<T *>(A->contents()), static_cast<T *>(B->contents()),
      static_cast<T *>(OUT->contents()), A->length() / sizeof(T), fn);
}

template <typename T, typename U>
bool verify_array_tolerant(const managed_ptr<MTL::Buffer> &A,
                           const managed_ptr<MTL::Buffer> &B,
                           const managed_ptr<MTL::Buffer> &OUT, U fn) {
  return verify_array_tolerant(
      static_cast<T *>(A->contents()), static_cast<T *>(B->contents()),
      static_cast<T *>(OUT->contents()), A->length() / sizeof(T), fn);
}

template <typename T>
void random(managed_ptr<MTL::Buffer> &buf) {
  auto p = static_cast<T *>(buf->contents());
  auto arr_len = buf->length() / sizeof(T);
  for (size_t i = 0; i < arr_len; i++) {
    p[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
  }
}

TEST_CASE("testing basic operations") {
  auto dev = managed(MTL::CreateSystemDefaultDevice());
  auto mtl = metal(dev.get());

  size_t arr_len = 60 * 180 * 10000;
  size_t buf_len = arr_len * sizeof(float);

  auto A = mtl.newBuffer(buf_len);
  auto B = mtl.newBuffer(buf_len);
  auto OUT = mtl.newBuffer(buf_len);

  random<float>(A);
  random<float>(B);

  mtl.compute<float>(A.get(), 0, A->length(), B.get(), 0, B->length(), OUT.get(),
                     0, OUT->length(), Operation::Add);
  CHECK(verify_array<float>(A, B, OUT, [](auto a, auto b) { return a + b; }));

  mtl.compute<float>(A.get(), 0, A->length(), B.get(), 0, B->length(), OUT.get(),
                     0, OUT->length(), Operation::Sub);
  CHECK(verify_array<float>(A, B, OUT, [](auto a, auto b) { return a - b; }));

  mtl.compute<float>(A.get(), 0, A->length(), B.get(), 0, B->length(), OUT.get(),
                     0, OUT->length(), Operation::Mul);
  CHECK(verify_array<float>(A, B, OUT, [](auto a, auto b) { return a * b; }));

  mtl.compute<float>(A.get(), 0, A->length(), B.get(), 0, B->length(), OUT.get(),
                     0, OUT->length(), Operation::Div);
  CHECK(verify_array_tolerant<float>(A, B, OUT,
                                     [](auto a, auto b) { return a / b; }));
}
