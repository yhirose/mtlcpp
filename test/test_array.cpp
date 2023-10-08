#include <array.h>

#include "doctest.h"
#include "utils.h"

using namespace mtlcpp;

template <typename T, typename U>
bool verify(const Array<T> &A, const Array<T> &B, const Array<T> &OUT, U fn) {
  return verify(A.data(), B.data(), OUT.data(), A.length(), fn);
}

template <typename T, typename U>
bool verify_tolerant(const Array<T> &A, const Array<T> &B, const Array<T> &OUT,
                     U fn) {
  return verify_tolerant(A.data(), B.data(), OUT.data(), A.length(), fn);
}

TEST_CASE("testing size of Array object") {
  Array<int> a{1, 2, 3, 4};
  CHECK(sizeof(a) == 24);
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

TEST_CASE("testing with container") {
  std::vector<int> v{1, 2, 3, 4};
  Array<int> a{v};
  CHECK(a.length() == 4);

  Array<int> b{std::vector<int>{1, 2, 3, 4}};
  CHECK(b.length() == 4);
}

TEST_CASE("testing with ranges") {
  Array<int> a{std::views::iota(1) | std::views::take(10)};
  CHECK(a.length() == 10);
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
}

TEST_CASE("testing bounds check") {
  Array<int> a{std::views::iota(0) | std::views::take(10)};
  CHECK(a[9] == 9);
  CHECK_THROWS_WITH_AS(a[10], "array: Index is out of bounds.",
                       std::runtime_error);
}

TEST_CASE("testing range-for") {
  auto a = zeros<float>(8);
  for (auto &x : a) {
    x = 1;
  }
  CHECK(ones<float>(8) == a);
}

TEST_CASE("testing arithmatic binary operations") {
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
  CHECK(out.length() == kLength);
  CHECK(
      verify_tolerant<float>(a, b, out, [](auto a, auto b) { return a / b; }));
}

TEST_CASE("testing arithmatic binary operation errors") {
  auto a = random<float>(4);
  auto b = random<float>(8);
  CHECK(a != b);

  CHECK_THROWS_WITH_AS(a + b, "array: Invalid operation.", std::runtime_error);
}

TEST_CASE("testing arithmatic operations") {
  Array<int> a{1, 2, 3, 4, 5, 6};
  CHECK(a.sum() == 21);

  CHECK(a.mean() == 3);
  CHECK(a.mean<double>() == 3.5);
}
