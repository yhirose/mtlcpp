#include <array.h>

#include <iostream>

#include "doctest.h"
#include "utils.h"

using namespace mtl;

template <typename T, typename U>
bool verify(const array<T> &A, const array<T> &B, const array<T> &OUT, U fn) {
  return verify(A.data(), B.data(), OUT.data(), A.length(), fn);
}

template <typename T, typename U>
bool verify_tolerant(const array<T> &A, const array<T> &B, const array<T> &OUT,
                     U fn) {
  return verify_tolerant(A.data(), B.data(), OUT.data(), A.length(), fn);
}

//------------------------------------------------------------------------------

TEST_CASE("vector: size") {
  auto v = vec::vector<int>(3);
  CHECK(v.length() == 3);
  CHECK(v.shape() == shape_type{3});
  CHECK(v.shape(0) == 3);
  CHECK(v.dimension() == 1);
}

TEST_CASE("vector: initializer") {
  auto v = vec::vector<int>({1, 2, 3, 4});
  CHECK(v.length() == 4);
}

TEST_CASE("vector: container") {
  std::vector<int> c{1, 2, 3, 4};
  auto v = vec::vector<int>(c.size(), c);
  CHECK(v.length() == 4);
}

TEST_CASE("vector ranges") {
  auto v = vec::vector<int>(10, std::views::iota(1) | std::views::take(10));
  CHECK(v.length() == 10);
}

TEST_CASE("vector: `clone`") {
  auto a = vec::ones<float>(8);
  auto b = a;
  a.zeros();
  CHECK(a == b);

  b = a.clone();
  a.ones();
  CHECK(a != b);
}

TEST_CASE("vector: assignment operator") {
  auto v = vec::zeros<float>(8);
  for (size_t i = 0; i < v.length(); i++) {
    v[i] = 1;
  }
  CHECK(vec::ones<float>(8) == v);
}

TEST_CASE("vector: bounds check") {
  auto v = vec::vector<int>(10, std::views::iota(0) | std::views::take(10));
  CHECK(v[9] == 9);
  CHECK_THROWS_WITH_AS(v[10], "array: Index is out of bounds.",
                       std::runtime_error);
}

TEST_CASE("vector: range-for") {
  auto v = vec::zeros<float>(8);
  for (auto &x : v) {
    x = 1;
  }
  CHECK(vec::ones<float>(8) == v);
}

TEST_CASE("vector: arithmatic binary operations") {
  constexpr size_t length = 16;

  auto a = vec::random<float>(length);
  auto b = vec::random<float>(length);

  auto out = a + b;
  CHECK(out.length() == length);
  CHECK(verify<float>(a, b, out, [](auto a, auto b) { return a + b; }));

  out = a - b;
  CHECK(out.length() == length);
  CHECK(verify<float>(a, b, out, [](auto a, auto b) { return a - b; }));

  out = a * b;
  CHECK(out.length() == length);
  CHECK(verify<float>(a, b, out, [](auto a, auto b) { return a * b; }));

  out = a / b;
  CHECK(out.length() == length);
  CHECK(
      verify_tolerant<float>(a, b, out, [](auto a, auto b) { return a / b; }));
}

TEST_CASE("vector: arithmatic binary operation errors") {
  auto a = vec::random<float>(4);
  auto b = vec::random<float>(8);
  CHECK(a != b);

  CHECK_THROWS_WITH_AS(a + b, "array: Invalid operation.", std::runtime_error);
}

TEST_CASE("vector: arithmatic functions") {
  auto a = vec::vector<int>({1, 2, 3, 4, 5, 6});
  CHECK(a.sum() == 21);

  CHECK(a.mean() == 3);
  CHECK(a.mean<double>() == 3.5);
}

//------------------------------------------------------------------------------

TEST_CASE("matrix: size") {
  auto m = mat::matrix<int>(3, 4);
  CHECK(m.length() == 12);
  CHECK(m.shape() == shape_type{3, 4});
  CHECK(m.shape(0) == 3);
  CHECK(m.shape(1) == 4);
  CHECK(m.dimension() == 2);
}

TEST_CASE("matrix: ranges") {
  auto m = mat::matrix<int>(3, 4, std::views::iota(1) | std::views::take(12));

  size_t i = 0;
  for (size_t row = 0; row < m.shape(0); row++) {
    for (size_t col = 0; col < m.shape(1); col++) {
      CHECK(m(row, col) == m[i]);
      i++;
    }
  }

  std::stringstream ss;
  ss << m;

  CHECK(R"([[1 2 3 4]
 [5 6 7 8]
 [9 10 11 12]])" == ss.str());
}

TEST_CASE("matrix: arithmatic binary operations") {
  auto r = std::views::iota(1) | std::views::take(12);

  auto a = mat::matrix<int>(3, 4, r);
  auto b = mat::matrix<int>(3, 4, r);
  auto out = a * b;
  out = out + 1;

  auto expected = mat::matrix<int>(
      3, 4, r | std::views::transform([](auto x) { return x * x + 1; }));

  CHECK(expected == out);
}
