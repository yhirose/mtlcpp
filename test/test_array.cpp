#include <array.h>

#include <iostream>

#include "doctest.h"
#include "utils.h"

using namespace mtl;

template <typename T, typename U>
bool verify_array(const array<T> &A, const array<T> &B, const array<T> &OUT,
                  U fn) {
  return verify_array(A.buffer_data(), B.buffer_data(), OUT.buffer_data(),
                      A.buffer_element_count(), fn);
}

template <typename T, typename U>
bool verify_array_tolerant(const array<T> &A, const array<T> &B,
                           const array<T> &OUT, U fn) {
  return verify_array_tolerant(A.buffer_data(), B.buffer_data(),
                               OUT.buffer_data(), A.buffer_element_count(), fn);
}

//------------------------------------------------------------------------------

TEST_CASE("scalar: size") {
  auto s = array<int>(0);
  CHECK(s.element_count() == 1);
  CHECK_THROWS_WITH_AS(s.length(), "array: cannot call with a scalar value.",
                       std::runtime_error);
  CHECK(s.dimension() == 0);
  CHECK(s.shape() == shape_type{});
}

//------------------------------------------------------------------------------

TEST_CASE("vector: size") {
  auto v = vector<int>(3);
  CHECK(v.element_count() == 3);
  CHECK(v.length() == 3);
  CHECK(v.dimension() == 1);
  CHECK(v.shape() == shape_type{3});
  CHECK(v.shape()[0] == 3);
}

TEST_CASE("vector: initializer") {
  auto v = array<int>{1, 2, 3, 4};
  CHECK(v.element_count() == 4);
}

TEST_CASE("vector: container") {
  std::vector<int> c{1, 2, 3, 4};
  auto v = vector<int>(c.size(), c);
  CHECK(v.element_count() == 4);
}

TEST_CASE("vector ranges") {
  auto v = vector<int>(10, std::views::iota(1) | std::views::take(10));
  CHECK(v.element_count() == 10);
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
  for (size_t i = 0; i < v.element_count(); i++) {
    v.at(i) = 1;
  }
  CHECK(ones<float>(8) == v);
}

TEST_CASE("vector: bounds check") {
  auto v = vector<int>(10, std::views::iota(0) | std::views::take(10));
  CHECK(v.at(9) == 9);
  CHECK_THROWS_WITH_AS(v.at(10), "array: index is out of bounds.",
                       std::runtime_error);
}

TEST_CASE("vector: range-for") {
  auto v = zeros<float>(8);
  std::fill(v.buffer_data(), v.buffer_data() + v.buffer_element_count(), 1);
  CHECK(ones<float>(8) == v);
}

TEST_CASE("vector: arithmatic operations") {
  constexpr size_t element_count = 16;

  auto a = random<float>(element_count);
  auto b = random<float>(element_count);

  auto out = a + b;
  CHECK(out.element_count() == element_count);
  CHECK(verify_array<float>(a, b, out, [](auto a, auto b) { return a + b; }));

  out = a - b;
  CHECK(out.element_count() == element_count);
  CHECK(verify_array<float>(a, b, out, [](auto a, auto b) { return a - b; }));

  out = a * b;
  CHECK(out.element_count() == element_count);
  CHECK(verify_array<float>(a, b, out, [](auto a, auto b) { return a * b; }));

  out = a / b;
  CHECK(out.element_count() == element_count);
  CHECK(verify_array_tolerant<float>(a, b, out,
                                     [](auto a, auto b) { return a / b; }));
}

TEST_CASE("vector: arithmatic operation errors") {
  auto a = random<float>(4);
  auto b = random<float>(8);
  CHECK(a != b);

  CHECK_THROWS_WITH_AS(a + b, "array: invalid operation.", std::runtime_error);
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
  CHECK(m.element_count() == 12);
  CHECK(m.shape() == shape_type{3, 4});
  CHECK(m.shape()[0] == 3);
  CHECK(m.shape()[1] == 4);
  CHECK(m.dimension() == 2);
}

TEST_CASE("matrix: container") {
  auto m1 = array<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  CHECK(m1.element_count() == 12);
  CHECK(m1.dimension() == 1);
  CHECK(m1.shape() == shape_type{12});
  CHECK(m1.strides() == strides_type{1});

  auto m2 = array<int>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  CHECK(m2.element_count() == 12);
  CHECK(m2.dimension() == 2);
  CHECK(m2.shape() == shape_type{3, 4});
  CHECK(m2.strides() == strides_type{4, 1});

  auto m3 = array<int>{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
  CHECK(m3.element_count() == 12);
  CHECK(m3.dimension() == 3);
  CHECK(m3.shape() == shape_type{2, 2, 3});
  CHECK(m3.strides() == strides_type{6, 3, 1});

  CHECK_THROWS_WITH_AS(
      (array<int>{{{1, 2, 3}, {4, 5}}, {{7, 8, 9}, {10, 11, 12}}}),
      "array: invalid initializer list.", std::runtime_error);
}

TEST_CASE("matrix: ranges") {
  auto m = matrix<int>(3, 4, std::views::iota(1) | std::views::take(12));

  size_t i = 0;
  for (size_t row = 0; row < m.shape()[0]; row++) {
    for (size_t col = 0; col < m.shape()[1]; col++) {
      CHECK(m.at(row, col) == m.at(i));
      i++;
    }
  }

  std::stringstream ss;
  ss << m;

  CHECK(R"([[1 2 3 4]
 [5 6 7 8]
 [9 10 11 12]])" == ss.str());
}

TEST_CASE("matrix: arithmatic operations") {
  auto r = itoa(12);
  auto a = matrix<int>(3, 4, r);
  auto b = matrix<int>(3, 4, r);
  CHECK(a + b == array<int>{{2, 4, 6, 8}, {10, 12, 14, 16}, {18, 20, 22, 24}});
  CHECK(a - b == array<int>{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
  CHECK(a * b ==
        array<int>{{1, 4, 9, 16}, {25, 36, 49, 64}, {81, 100, 121, 144}});
  CHECK(a / b == array<int>{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}});
}

TEST_CASE("matrix: arithmatic operations with scalar") {
  auto a = array<float>{{1, 2}, {3, 4}};
  CHECK(a + 1 == array<float>{{2, 3}, {4, 5}});
  CHECK(a - 1 == array<float>{{0, 1}, {2, 3}});
  CHECK(a * 2 == array<float>{{2, 4}, {6, 8}});
  CHECK(a / 2 == array<float>{{0.5, 1}, {1.5, 2}});
  CHECK(1 + a == array<float>{{2, 3}, {4, 5}});
  CHECK(1 - a == array<float>{{0, -1}, {-2, -3}});
  CHECK(2 * a == array<float>{{2, 4}, {6, 8}});
  CHECK(2 / a == array<float>{{2, 1}, {2.0 / 3.0, 0.5}});
}

TEST_CASE("matrix: v*v `dot` operation") {
  auto a = vector<int>(4, itoa(4));
  auto b = vector<int>(4, itoa(4));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{});

  auto expected = array<int>(30);
  CHECK(out == expected);
}

TEST_CASE("matrix: m*m `dot` operation") {
  auto a = matrix<int>(3, 4, itoa(12));
  auto b = matrix<int>(4, 2, itoa(8));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{3, 2});

  auto expected = matrix<int>(3, 2);
  expected.set({50, 60, 114, 140, 178, 220});
  CHECK(out == expected);
}

TEST_CASE("matrix: v*m `dot` operation") {
  auto a = vector<int>(4, itoa(4));
  auto b = matrix<int>(4, 2, itoa(8));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{2});

  auto expected = vector<int>(2);
  expected.set({50, 60});
  CHECK(out == expected);
}

TEST_CASE("matrix: m*v `dot` operation") {
  auto a = matrix<int>(2, 4, itoa(8));
  auto b = vector<int>(4, itoa(4));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{2});

  auto expected = vector<int>(2);
  expected.set({30, 70});
  CHECK(out == expected);
}

TEST_CASE("matrix: transpose") {
  auto v = array<int>{1, 2, 3, 4};
  auto vT = v.transpose();
  CHECK(vT.element_count() == 4);
  CHECK(vT.dimension() == 2);
  CHECK(vT.shape() == shape_type{1, 4});
  auto vT_expected = array<int>{1, 2, 3, 4};
  vT_expected.reshape({1, 4});
  CHECK(vT == vT_expected);

  auto vT2 = vT.transpose();
  CHECK(vT2.element_count() == 4);
  CHECK(vT2.dimension() == 1);
  CHECK(vT2.shape() == shape_type{4});
  auto vT2_expected = array<int>{1, 2, 3, 4};
  CHECK(vT2 == vT2_expected);

  auto m2 = array<int>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  auto m2T = m2.transpose();
  CHECK(m2T.element_count() == 12);
  CHECK(m2T.dimension() == 2);
  CHECK(m2T.shape() == shape_type{4, 3});
  CHECK(m2T == array<int>{{1, 5, 9}, {2, 6, 10}, {3, 7, 11}, {4, 8, 12}});

  auto m2T2 = m2T.transpose();
  CHECK(m2T2.element_count() == m2.element_count());
  CHECK(m2T2.dimension() == m2.dimension());
  CHECK(m2T2.shape() == m2.shape());
  CHECK(m2T2 == m2);

  auto m3 = array<int>{{{1, 2, 3, 4}, {5, 6, 7, 8}},
                       {{9, 10, 11, 12}, {13, 14, 15, 16}}};
  CHECK(m3.element_count() == 16);
  CHECK(m3.dimension() == 3);
  CHECK(m3.shape() == shape_type{2, 2, 4});

  auto m3T = m3.transpose();
  CHECK(m3T.element_count() == 16);
  CHECK(m3T.dimension() == 3);
  CHECK(m3T.shape() == shape_type{4, 2, 2});

  auto m3T2 = m3T.transpose();
  CHECK(m3T2.element_count() == m3.element_count());
  CHECK(m3T2.dimension() == m3.dimension());
  CHECK(m3T2.shape() == m3.shape());
  CHECK(m3T2 == m3);
}

TEST_CASE("matrix: broadcast") {
  auto a = array<int>{{1, 2, 3}, {4, 5, 6}};
  auto b = a.broadcast({3, 2, 3});

  CHECK(b == array<int>{{{1, 2, 3}, {4, 5, 6}},
                        {{1, 2, 3}, {4, 5, 6}},
                        {{1, 2, 3}, {4, 5, 6}}});

  CHECK(b.element_count() == 18);
  CHECK(b.buffer_element_count() == 6);
  CHECK(b.buffer_bytes() == 6 * sizeof(int));

  CHECK(b.at(0) == 1);
  CHECK(b.at(b.element_count() - 1) == 6);

  CHECK(b.at(0, 0, 0) == 1);
  CHECK(b.at(1, 1, 0) == 4);
  CHECK(b.at(2, 1, 2) == 6);

  CHECK(b.strides().size() == 3);
  CHECK(b.strides()[0] == 0);
  CHECK(b.strides()[1] == 3);
  CHECK(b.strides()[2] == 1);
}

TEST_CASE("matrix: arithmatic operations with broadcast") {
  auto a_2_3 = array<int>{{1, 2, 3}, {4, 5, 6}};
  auto a_2_2_3 = array<int>{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};

  auto b = array<int>(1);
  auto b_3 = array<int>{1, 2, 3};
  auto b_2_3 = array<int>{{1, 2, 3}, {4, 5, 6}};

  CHECK(a_2_3 + b == array<int>{{2, 3, 4}, {5, 6, 7}});
  CHECK(a_2_2_3 + b ==
        array<int>{{{2, 3, 4}, {5, 6, 7}}, {{8, 9, 10}, {11, 12, 13}}});
  CHECK(a_2_3 + b_3 == array<int>{{2, 4, 6}, {5, 7, 9}});
  CHECK(a_2_2_3 + b_3 ==
        array<int>{{{2, 4, 6}, {5, 7, 9}}, {{8, 10, 12}, {11, 13, 15}}});
  CHECK(a_2_2_3 + b_2_3 ==
        array<int>{{{2, 4, 6}, {8, 10, 12}}, {{8, 10, 12}, {14, 16, 18}}});

  CHECK(b + a_2_3 == array<int>{{2, 3, 4}, {5, 6, 7}});
  CHECK(b + a_2_2_3 ==
        array<int>{{{2, 3, 4}, {5, 6, 7}}, {{8, 9, 10}, {11, 12, 13}}});
  CHECK(b_3 + a_2_3 == array<int>{{2, 4, 6}, {5, 7, 9}});
  CHECK(b_3 + a_2_2_3 ==
        array<int>{{{2, 4, 6}, {5, 7, 9}}, {{8, 10, 12}, {11, 13, 15}}});
  CHECK(b_2_3 + a_2_2_3 ==
        array<int>{{{2, 4, 6}, {8, 10, 12}}, {{8, 10, 12}, {14, 16, 18}}});
}

TEST_CASE("matrix: test") {
  auto t = array<int>{
      {{1, 2, 3}, {4, 5, 6}},
      {{7, 8, 9}, {10, 11, 12}},
      {{13, 14, 15}, {16, 17, 18}},
  };

  CHECK_THROWS_WITH_AS(t(3), "array: row is out of bounds.",
                       std::runtime_error);

  auto m = t(1);
  auto v = m(1);
  auto s = v(1);

  CHECK(m == array<int>{{7, 8, 9}, {10, 11, 12}});
  CHECK(v == array<int>{10, 11, 12});
  CHECK(s == array<int>(11));

  s.at() += 100;

  CHECK(t == array<int>{{{1, 2, 3}, {4, 5, 6}},
                        {{7, 8, 9}, {10, 111, 12}},
                        {{13, 14, 15}, {16, 17, 18}}});
  CHECK(m == array<int>{{7, 8, 9}, {10, 111, 12}});
  CHECK(v == array<int>{10, 111, 12});
  CHECK(s == array<int>(111));

  m.zeros();

  CHECK(t == array<int>{{{1, 2, 3}, {4, 5, 6}},
                        {{0, 0, 0}, {0, 0, 0}},
                        {{13, 14, 15}, {16, 17, 18}}});
  CHECK(m == array<int>{{0, 0, 0}, {0, 0, 0}});
  CHECK(v == array<int>{0, 0, 0});
  CHECK(s == array<int>(0));
}
