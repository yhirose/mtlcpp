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
  auto v = vector<int>(3);
  CHECK(v.length() == 3);
  CHECK(v.dimension() == 1);
  CHECK(v.shape() == shape_type{3});
  CHECK(v.shape(0) == 3);
}

TEST_CASE("vector: initializer") {
  auto v = array<int>{1, 2, 3, 4};
  CHECK(v.length() == 4);
}

TEST_CASE("vector: container") {
  std::vector<int> c{1, 2, 3, 4};
  auto v = vector<int>(c.size(), c);
  CHECK(v.length() == 4);
}

TEST_CASE("vector ranges") {
  auto v = vector<int>(10, std::views::iota(1) | std::views::take(10));
  CHECK(v.length() == 10);
}

TEST_CASE("vector: `clone`") {
  auto a = ones<float>(8);
  auto b = a;
  a.zeros();
  CHECK(a == b);

  b = a.clone();
  a.ones();
  CHECK(a != b);
}

TEST_CASE("vector: assignment operator") {
  auto v = zeros<float>(8);
  for (size_t i = 0; i < v.length(); i++) {
    v[i] = 1;
  }
  CHECK(ones<float>(8) == v);
}

TEST_CASE("vector: bounds check") {
  auto v = vector<int>(10, std::views::iota(0) | std::views::take(10));
  CHECK(v[9] == 9);
  CHECK_THROWS_WITH_AS(v[10], "array: Index is out of bounds.",
                       std::runtime_error);
}

TEST_CASE("vector: range-for") {
  auto v = zeros<float>(8);
  for (auto &x : v) {
    x = 1;
  }
  CHECK(ones<float>(8) == v);
}

TEST_CASE("vector: arithmatic binary operations") {
  constexpr size_t length = 16;

  auto a = random<float>(length);
  auto b = random<float>(length);

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
  auto a = random<float>(4);
  auto b = random<float>(8);
  CHECK(a != b);

  CHECK_THROWS_WITH_AS(a + b, "array: Invalid operation.", std::runtime_error);
}

TEST_CASE("vector: arithmatic functions") {
  auto a = array<int>{1, 2, 3, 4, 5, 6};
  CHECK(a.sum() == 21);

  CHECK(a.mean() == 3.5);
  CHECK(a.mean<int>() == 3);
}

//------------------------------------------------------------------------------

TEST_CASE("matrix: size") {
  auto m = matrix<int>(3, 4);
  CHECK(m.length() == 12);
  CHECK(m.shape() == shape_type{3, 4});
  CHECK(m.shape(0) == 3);
  CHECK(m.shape(1) == 4);
  CHECK(m.dimension() == 2);
}

TEST_CASE("matrix: container") {
  auto m1 = array<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  CHECK(m1.length() == 12);
  CHECK(m1.dimension() == 1);
  CHECK(m1.shape() == shape_type{12});

  auto m2 = array<int>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  CHECK(m2.length() == 12);
  CHECK(m2.dimension() == 2);
  CHECK(m2.shape() == shape_type{3,4});

  auto m3 = array<int>{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
  CHECK(m3.length() == 12);
  CHECK(m3.dimension() == 3);
  CHECK(m3.shape() == shape_type{2, 2, 3});
}

TEST_CASE("matrix: ranges") {
  auto m = matrix<int>(3, 4, std::views::iota(1) | std::views::take(12));

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
  auto r = itoa(12);

  auto a = matrix<int>(3, 4, r);
  auto b = matrix<int>(3, 4, r);
  auto out = a * b;
  out = out + 1;

  auto expected = matrix<int>(
      3, 4, r | std::views::transform([](auto x) { return x * x + 1; }));
  CHECK(expected == out);
}

TEST_CASE("matrix: v*v `dot` operation") {
  auto a = vector<int>(4, itoa(4));
  auto b = vector<int>(4, itoa(4));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{});

  auto expected = scalar<int>(30);
  CHECK(expected == out);
}

TEST_CASE("matrix: m*m `dot` operation") {
  auto a = matrix<int>(3, 4, itoa(12));
  auto b = matrix<int>(4, 2, itoa(8));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{3, 2});

  auto expected = matrix<int>(3, 2);
  expected.set({50, 60, 114, 140, 178, 220});
  CHECK(expected == out);
}

TEST_CASE("matrix: v*m `dot` operation") {
  auto a = vector<int>(4, itoa(4));
  auto b = matrix<int>(4, 2, itoa(8));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{2});

  auto expected = vector<int>(2);
  expected.set({50, 60});
  CHECK(expected == out);
}

TEST_CASE("matrix: m*v `dot` operation") {
  auto a = matrix<int>(2, 4, itoa(8));
  auto b = vector<int>(4, itoa(4));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{2});

  auto expected = vector<int>(2);
  expected.set({30, 70});
  CHECK(expected == out);
}
