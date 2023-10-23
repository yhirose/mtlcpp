#include <metal.h>

#include "doctest.h"

using namespace mtl;

template <typename T, typename U>
bool verify_array(const managed_ptr<MTL::Buffer> &A,
                  const managed_ptr<MTL::Buffer> &B,
                  const managed_ptr<MTL::Buffer> &OUT, U fn) {
  auto a = static_cast<T *>(A->contents());
  auto b = static_cast<T *>(B->contents());
  auto out = static_cast<T *>(OUT->contents());
  auto length = A->length() / sizeof(T);

  size_t err = 0;
  for (size_t i = 0; i < length; i++) {
    if (out[i] != fn(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

template <typename T, typename U>
bool verify_array_tolerant(const managed_ptr<MTL::Buffer> &A,
                           const managed_ptr<MTL::Buffer> &B,
                           const managed_ptr<MTL::Buffer> &OUT, U fn) {
  auto a = static_cast<T *>(A->contents());
  auto b = static_cast<T *>(B->contents());
  auto out = static_cast<T *>(OUT->contents());
  auto length = A->length() / sizeof(T);

  size_t err = 0;
  for (size_t i = 0; i < length; i++) {
    if (std::abs(out[i] - fn(a[i], b[i])) > 1e-3) {
      err++;
    }
  }
  if (err == 0) {
    return true;
  } else {
    auto ratio = static_cast<double>(err) / length * 100.0;
    return ratio < 0.001;
  }
}

template <typename T>
void random(managed_ptr<MTL::Buffer> &buf) {
  auto p = static_cast<T *>(buf->contents());
  auto arr_len = buf->length() / sizeof(T);
  for (size_t i = 0; i < arr_len; i++) {
    p[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
  }
}

//------------------------------------------------------------------------------

TEST_CASE("metal: basic operations") {
  auto dev = managed(MTL::CreateSystemDefaultDevice());
  auto mtl = metal(dev.get());

  size_t arr_len = 60 * 180 * 10000;
  size_t buf_len = arr_len * sizeof(float);

  auto A = mtl.newBuffer(buf_len);
  auto B = mtl.newBuffer(buf_len);
  auto OUT = mtl.newBuffer(buf_len);

  random<float>(A);
  random<float>(B);

  mtl.compute<float>(A.get(), 0, A->length(), B.get(), 0, B->length(),
                     OUT.get(), 0, OUT->length(), Operation::Add);
  CHECK(verify_array<float>(A, B, OUT, [](auto a, auto b) { return a + b; }));

  mtl.compute<float>(A.get(), 0, A->length(), B.get(), 0, B->length(),
                     OUT.get(), 0, OUT->length(), Operation::Sub);
  CHECK(verify_array<float>(A, B, OUT, [](auto a, auto b) { return a - b; }));

  mtl.compute<float>(A.get(), 0, A->length(), B.get(), 0, B->length(),
                     OUT.get(), 0, OUT->length(), Operation::Mul);
  CHECK(verify_array<float>(A, B, OUT, [](auto a, auto b) { return a * b; }));

  mtl.compute<float>(A.get(), 0, A->length(), B.get(), 0, B->length(),
                     OUT.get(), 0, OUT->length(), Operation::Div);
  CHECK(verify_array_tolerant<float>(A, B, OUT,
                                     [](auto a, auto b) { return a / b; }));
}
