#include <array.h>

#include "doctest.h"

using namespace mtlcpp;

template <typename T, typename U>
bool verify(const Array<T> &A, const Array<T> &B, const Array<T> &OUT, U fn) {
  for (size_t i = 0; i < A.length(); i++) {
    if (OUT[i] != fn(A[i], B[i])) {
      return false;
    }
  }
  return true;
}

TEST_CASE("testing length") {
  constexpr size_t arr_len = 8;

  auto u32 = random<uint32_t>(arr_len);
  CHECK(u32.length() == arr_len);

  auto f = random<float>(arr_len);
  CHECK(f.length() == arr_len);

  auto d = random<double>(arr_len);
  CHECK(d.length() == arr_len);
}

TEST_CASE("testing initializer") {
  Array<int> a{1, 2, 3, 4};
  CHECK(a.length() == 4);
}

TEST_CASE("testing copy operator") {
  auto a = ones<float>(8);
  auto b = a;
  a.zeros();
  CHECK(a == b);

  b = a.copy();
  a.ones();
  CHECK(a != b);
}

TEST_CASE("testing assignment operator") {
  auto a = zeros<float>(8);
  for (size_t i = 0; i < a.length(); i++) {
    a[i] = 1;
  }
  CHECK(ones<float>(8) == a);

  // TODO:: range check...
}

TEST_CASE("testing range-for") {
  auto a = zeros<float>(8);
  for (auto &x : a) {
    x = 1;
  }
  CHECK(ones<float>(8) == a);
}

TEST_CASE("testing arithmatic operations") {
  constexpr size_t kLength = 16;

  auto a = random<float>(kLength);
  auto b = random<float>(kLength);

  CHECK(a.length() == kLength);
  CHECK(b.length() == kLength);

  auto out = a + b;
  CHECK(out.length() == kLength);
  CHECK(verify<float>(a, b, out, [](auto a, auto b) { return a + b; }));

  out = a - b;
  CHECK(out.length() == kLength);
  CHECK(verify<float>(a, b, out, [](auto a, auto b) { return a - b; }));

  out = a * b;
  CHECK(out.length() == kLength);
  CHECK(verify<float>(a, b, out, [](auto a, auto b) { return a * b; }));

  out = a / b;
  // TODO:
  // CHECK(out.length() == kLength);
  // CHECK(verify<float>(a, b, out, [](auto a, auto b) { return a / b; }));
}

TEST_CASE("testing arithmatic operation errors") {
  auto a = random<float>(4);
  auto b = random<float>(8);
  CHECK(a != b);

  CHECK_THROWS_WITH_AS(a + b, "array: Invalid operation.", std::runtime_error);
}
