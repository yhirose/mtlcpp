#include <gpu.h>
#include <metal.h>

#include "doctest.h"
#include "utils.h"

using namespace mtlcpp;

template <typename T, typename U>
bool verify(const GPU::Memory &A, const GPU::Memory &B, const GPU::Memory &OUT,
            U fn) {
  return verify(A.data<T>(), B.data<T>(), OUT.data<T>(), A.length<T>(), fn);
}

template <typename T, typename U>
bool verify_tolerant(const GPU::Memory &A, const GPU::Memory &B,
                     const GPU::Memory &OUT, U fn) {
  return verify_tolerant(A.data<T>(), B.data<T>(), OUT.data<T>(), A.length<T>(),
                         fn);
}

template <typename T>
void random(GPU::Memory &m) {
  random(m.data<T>(), m.length<T>());
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
  CHECK(
      verify_tolerant<float>(A, B, OUT, [](auto a, auto b) { return a / b; }));
}
