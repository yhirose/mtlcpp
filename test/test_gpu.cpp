#include <gpu.h>
#include <metal.h>

#include "doctest.h"

using namespace mtlcpp;

template <typename T, typename U>
bool verify(const GPU::Memory &A, const GPU::Memory &B, const GPU::Memory &OUT,
            U fn) {
  auto a = A.data<T>();
  auto b = B.data<T>();
  auto out = OUT.data<T>();

  for (size_t i = 0; i < A.length<T>(); i++) {
    if (out[i] != fn(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

template <typename T>
void random(GPU::Memory &m) {
  auto p = m.data<T>();
  for (size_t i = 0; i < m.length<T>(); i++) {
    p[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
  }
}

TEST_CASE("testing basic operations") {
  size_t arr_len = 60 * 180 * 10000;

  auto A = GPU::allocate<float>(arr_len);
  auto B = GPU::allocate<float>(arr_len);
  auto OUT = GPU::allocate<float>(arr_len);

  random<float>(A);
  random<float>(B);

  GPU::compute<float>(A, B, OUT, ComputeType::ARRAY_ADD_F);
  CHECK(verify<float>(A, B, OUT, [](auto a, auto b) { return a + b; }));

  GPU::compute<float>(A, B, OUT, ComputeType::ARRAY_SUB_F);
  CHECK(verify<float>(A, B, OUT, [](auto a, auto b) { return a - b; }));

  GPU::compute<float>(A, B, OUT, ComputeType::ARRAY_MUL_F);
  CHECK(verify<float>(A, B, OUT, [](auto a, auto b) { return a * b; }));

  GPU::compute<float>(A, B, OUT, ComputeType::ARRAY_DIV_F);
  // TODO:
  // CHECK(verify<float>(A, B, OUT, [](auto a, auto b) { return a / b; }));
}
