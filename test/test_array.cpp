#include <mtlcpp.h>

#include <iostream>
#include <ranges>

#include "doctest.h"

using namespace mtl;

auto itoa(size_t size, size_t init = 1) {
  return std::views::iota(init) | std::views::take(size);
}

//------------------------------------------------------------------------------

TEST_CASE("array: scalar size") {
  auto s = array<int>(100);
  CHECK(s.element_count() == 1);
  CHECK_THROWS_WITH_AS(s.length(), "array: cannot call with a scalar value.",
                       std::runtime_error);
  CHECK(s.dimension() == 0);
  CHECK(s.shape() == shape_type{});
  CHECK(s.at() == 100);
}

//------------------------------------------------------------------------------

TEST_CASE("array: vector size") {
  auto v = empty<int>({3});
  CHECK(v.element_count() == 3);
  CHECK(v.length() == 3);
  CHECK(v.dimension() == 1);
  CHECK(v.shape() == shape_type{3});
  CHECK(v.shape()[0] == 3);
}

TEST_CASE("array: vector initializer") {
  auto v = array<int>{1, 2, 3, 4};
  CHECK(v.element_count() == 4);
}

TEST_CASE("vector: container") {
  std::vector<int> a{1, 2, 3, 4};

  auto v1 = array<int>({a.size() - 1}, a);
  CHECK(v1.element_count() == 3);
  CHECK(array_equal(v1, {1, 2, 3}));

  auto v2 = array<int>({a.size() + 1}, a);
  CHECK(v2.element_count() == 5);
  CHECK(array_equal(v1, {1, 2, 3}));

  auto v3 = array<int>(a);
  CHECK(v3.element_count() == 4);
  CHECK(array_equal(v3, {1, 2, 3, 4}));
}

TEST_CASE("array: vector ranges") {
  auto v = array<int>({10}, std::views::iota(1) | std::views::take(10));
  CHECK(v.element_count() == 10);
  CHECK(array_equal(v, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
}

TEST_CASE("array: vector `clone`") {
  auto a = ones<float>({8});
  auto b = a;
  a.zeros();
  CHECK(array_equal(a, b));

  b = a.clone();
  a.ones();
  CHECK(!array_equal(a, b));
}

TEST_CASE("array: vector assignment operator") {
  auto v = zeros<float>({8});
  for (size_t i = 0; i < v.element_count(); i++) {
    v.at(i) = 1;
  }
  CHECK(array_equal(ones<float>({8}), v));
}

TEST_CASE("array: vector bounds check") {
  auto v = array<int>({10}, std::views::iota(0) | std::views::take(10));
  CHECK(v.at(9) == 9);
  CHECK_THROWS_WITH_AS(v.at(10), "array: index is out of bounds.",
                       std::runtime_error);
}

TEST_CASE("array: vector range-for") {
  auto v = zeros<float>({8});
  std::fill(v.buffer_data(), v.buffer_data() + v.buffer_element_count(), 1);
  CHECK(array_equal(ones<float>({8}), v));
}

TEST_CASE("array: vector arithmatic operations") {
  constexpr size_t element_count = 16;

  auto a = array<float>{7.82637e-06, 0.131538,  0.755605,  0.45865,
                        0.532767,    0.218959,  0.0470446, 0.678865,
                        0.679296,    0.934693,  0.383502,  0.519416,
                        0.830965,    0.0345721, 0.0534616, 0.5297};

  auto b = array<float>{0.671149, 0.00769819, 0.383416,  0.0668422,
                        0.417486, 0.686773,   0.588977,  0.930436,
                        0.846167, 0.526929,   0.0919649, 0.653919,
                        0.415999, 0.701191,   0.910321,  0.762198};

  CHECK(allclose(a + b, {0.671157, 0.139236, 1.13902, 0.525492, 0.950253,
                         0.905732, 0.636021, 1.6093, 1.52546, 1.46162, 0.475467,
                         1.17334, 1.24696, 0.735763, 0.963782, 1.2919}));

  CHECK(allclose(
      a - b, {-0.671141, 0.12384, 0.372189, 0.391808, 0.115281, -0.467814,
              -0.541932, -0.251571, -0.166871, 0.407764, 0.291537, -0.134503,
              0.414966, -0.666619, -0.856859, -0.232498}));

  CHECK(allclose(
      a * b, {5.25266e-06, 0.0010126, 0.289711, 0.0306572, 0.222423, 0.150375,
              0.0277082, 0.63164, 0.574798, 0.492517, 0.0352687, 0.339656,
              0.345681, 0.0242416, 0.0486672, 0.403736}));

  CHECK(
      allclose(a / b, {1.16612e-05, 17.0869, 1.97072, 6.86168, 1.27613,
                       0.318823, 0.0798751, 0.72962, 0.802792, 1.77385, 4.17009,
                       0.794312, 1.99752, 0.0493048, 0.0587283, 0.694964}));
}

TEST_CASE("array: vector arithmatic operation errors") {
  auto a = random({4});
  auto b = random({8});
  CHECK(!array_equal(a, b));
  CHECK_THROWS_WITH_AS(a + b, "array: invalid operation.", std::runtime_error);
}

TEST_CASE("array: vector `pow` operation") {
  {
    auto a = array<int>{1, 2, 3};
    auto b = array<int>{2, 2, 2};
    CHECK(array_equal(a.pow(b), {1, 4, 9}));
    CHECK(array_equal(b.pow(a), {2, 4, 8}));
  }
  {
    auto a = array<float>{1.0, 2.0, 3.0};
    auto b = array<float>{2.0, 2.0, 2.0};
    CHECK(allclose(a.pow(b), {1.0, 4.0, 9.0}));
    CHECK(allclose(b.pow(a), {2.0, 4.0, 8.0}));
  }
}

//------------------------------------------------------------------------------

TEST_CASE("array: matrix size") {
  auto m = empty<int>({3, 4});
  CHECK(m.element_count() == 12);
  CHECK(m.shape() == shape_type{3, 4});
  CHECK(m.shape()[0] == 3);
  CHECK(m.shape()[1] == 4);
  CHECK(m.dimension() == 2);
}

TEST_CASE("array: matrix container") {
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

TEST_CASE("array: matrix ranges") {
  auto m = array<int>({3, 4}, std::views::iota(1) | std::views::take(12));

  size_t i = 0;
  for (size_t row = 0; row < m.shape()[0]; row++) {
    for (size_t col = 0; col < m.shape()[1]; col++) {
      CHECK(m.at(row, col) == m.at(i));
      i++;
    }
  }

  CHECK(array_equal(m, {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}));
}

TEST_CASE("array: matrix arithmatic operations") {
  auto r = itoa(12);
  auto a = array<int>({3, 4}, r);
  auto b = array<int>({3, 4}, r);
  CHECK(array_equal(a + b, {{2, 4, 6, 8}, {10, 12, 14, 16}, {18, 20, 22, 24}}));
  CHECK(array_equal(a - b, {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}));
  CHECK(array_equal(a * b,
                    {{1, 4, 9, 16}, {25, 36, 49, 64}, {81, 100, 121, 144}}));
  CHECK(array_equal(a / b, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}));
}

TEST_CASE("array: matrix arithmatic operations with scalar") {
  auto a = array<float>{{1, 2}, {3, 4}};
  CHECK(array_equal(a + 1, {{2, 3}, {4, 5}}));
  CHECK(array_equal(a - 1, {{0, 1}, {2, 3}}));
  CHECK(array_equal(a * 2, {{2, 4}, {6, 8}}));
  CHECK(array_equal(a / 2, {{0.5, 1}, {1.5, 2}}));
  CHECK(array_equal(1 + a, {{2, 3}, {4, 5}}));
  CHECK(array_equal(1 - a, {{0, -1}, {-2, -3}}));
  CHECK(array_equal(2 * a, {{2, 4}, {6, 8}}));
  CHECK(array_equal(2 / a, {{2, 1}, {2.0 / 3.0, 0.5}}));
}

TEST_CASE("array: matrix v*v `dot` operation") {
  auto a = array<int>({4}, itoa(4));
  auto b = array<int>({4}, itoa(4));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{});
  CHECK(array_equal(out, array<int>(30)));
}

TEST_CASE("array: matrix m*m `dot` operation") {
  auto a = array<int>({3, 4}, itoa(12));
  auto b = array<int>({4, 2}, itoa(8));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{3, 2});
  CHECK(array_equal(out, {{50, 60}, {114, 140}, {178, 220}}));
}

TEST_CASE("array: matrix v*m `dot` operation") {
  auto a = array<int>({4}, itoa(4));
  auto b = array<int>({4, 2}, itoa(8));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{2});
  CHECK(array_equal(out, {50, 60}));
}

TEST_CASE("array: matrix m*v `dot` operation") {
  auto a = array<int>({2, 4}, itoa(8));
  auto b = array<int>({4}, itoa(4));
  auto out = a.dot(b);
  CHECK(out.shape() == shape_type{2});
  CHECK(array_equal(out, {30, 70}));
}

TEST_CASE("array: matrix transpose") {
  auto v = array<int>{1, 2, 3, 4};
  auto vT = v.transpose();
  CHECK(vT.element_count() == 4);
  CHECK(vT.dimension() == 2);
  CHECK(vT.shape() == shape_type{1, 4});
  CHECK(array_equal(vT, {{1, 2, 3, 4}}));

  auto vT2 = vT.transpose();
  CHECK(vT2.element_count() == 4);
  CHECK(vT2.dimension() == 1);
  CHECK(vT2.shape() == shape_type{4});
  CHECK(array_equal(vT2, {1, 2, 3, 4}));

  auto m2 = array<int>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  auto m2T = m2.transpose();
  CHECK(m2T.element_count() == 12);
  CHECK(m2T.dimension() == 2);
  CHECK(m2T.shape() == shape_type{4, 3});
  CHECK(array_equal(m2T, {{1, 5, 9}, {2, 6, 10}, {3, 7, 11}, {4, 8, 12}}));

  auto m2T2 = m2T.transpose();
  CHECK(m2T2.element_count() == m2.element_count());
  CHECK(m2T2.dimension() == m2.dimension());
  CHECK(m2T2.shape() == m2.shape());
  CHECK(array_equal(m2T2, m2));

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
  CHECK(array_equal(m3T2, m3));
}

TEST_CASE("array: matrix broadcast") {
  auto a = array<int>{{1, 2, 3}, {4, 5, 6}};
  auto b = a.broadcast({3, 2, 3});

  CHECK(array_equal(b, {{{1, 2, 3}, {4, 5, 6}},
                        {{1, 2, 3}, {4, 5, 6}},
                        {{1, 2, 3}, {4, 5, 6}}}));

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

TEST_CASE("array: matrix arithmatic operations with broadcast") {
  auto a_2_3 = array<int>{{1, 2, 3}, {4, 5, 6}};
  auto a_2_2_3 = array<int>{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};

  auto b = array<int>(1);
  auto b_3 = array<int>{1, 2, 3};
  auto b_2_3 = array<int>{{1, 2, 3}, {4, 5, 6}};

  CHECK(array_equal(a_2_3 + b, {{2, 3, 4}, {5, 6, 7}}));
  CHECK(array_equal(a_2_2_3 + b,
                    {{{2, 3, 4}, {5, 6, 7}}, {{8, 9, 10}, {11, 12, 13}}}));
  CHECK(array_equal(a_2_3 + b_3, {{2, 4, 6}, {5, 7, 9}}));
  CHECK(array_equal(a_2_2_3 + b_3,
                    {{{2, 4, 6}, {5, 7, 9}}, {{8, 10, 12}, {11, 13, 15}}}));
  CHECK(array_equal(a_2_2_3 + b_2_3,
                    {{{2, 4, 6}, {8, 10, 12}}, {{8, 10, 12}, {14, 16, 18}}}));

  CHECK(array_equal(b + a_2_3, {{2, 3, 4}, {5, 6, 7}}));
  CHECK(array_equal(b + a_2_2_3,
                    {{{2, 3, 4}, {5, 6, 7}}, {{8, 9, 10}, {11, 12, 13}}}));
  CHECK(array_equal(b_3 + a_2_3, {{2, 4, 6}, {5, 7, 9}}));
  CHECK(array_equal(b_3 + a_2_2_3,
                    {{{2, 4, 6}, {5, 7, 9}}, {{8, 10, 12}, {11, 13, 15}}}));
  CHECK(array_equal(b_2_3 + a_2_2_3,
                    {{{2, 4, 6}, {8, 10, 12}}, {{8, 10, 12}, {14, 16, 18}}}));
}

TEST_CASE("array: matrix slice") {
  auto t = array<int>{
      {{1, 2, 3}, {4, 5, 6}},
      {{7, 8, 9}, {10, 11, 12}},
      {{13, 14, 15}, {16, 17, 18}},
  };

  CHECK_THROWS_WITH_AS(t[3], "array: row is out of bounds.",
                       std::runtime_error);

  auto m = t[1];
  auto v = m[1];
  auto s = v[1];

  CHECK(array_equal(m, {{7, 8, 9}, {10, 11, 12}}));
  CHECK(array_equal(v, {10, 11, 12}));
  CHECK(array_equal(s, array<int>(11)));

  s.at() += 100;

  CHECK(array_equal(t, {{{1, 2, 3}, {4, 5, 6}},
                        {{7, 8, 9}, {10, 111, 12}},
                        {{13, 14, 15}, {16, 17, 18}}}));
  CHECK(array_equal(m, {{7, 8, 9}, {10, 111, 12}}));
  CHECK(array_equal(v, {10, 111, 12}));
  CHECK(array_equal(s, array<int>(111)));

  m.zeros();

  CHECK(array_equal(t, {{{1, 2, 3}, {4, 5, 6}},
                        {{0, 0, 0}, {0, 0, 0}},
                        {{13, 14, 15}, {16, 17, 18}}}));
  CHECK(array_equal(m, {{0, 0, 0}, {0, 0, 0}}));
  CHECK(array_equal(v, {0, 0, 0}));
  CHECK(array_equal(s, array<int>(0)));
}

//------------------------------------------------------------------------------

TEST_CASE("array: aggregate functions") {
  auto v = array<int>{1, 2, 3, 4, 5, 6};

  auto t = array<int>{
      {{1, 2, 3}, {4, 5, 6}},
      {{7, 8, 9}, {10, 11, 12}},
      {{13, 14, 15}, {16, 17, 18}},
  };

  CHECK(v.min() == 1);
  CHECK(v.max() == 6);
  CHECK(t.min() == 1);
  CHECK(t.max() == 18);

  CHECK(v.sum() == 21);
  CHECK(t.sum() == 171);
  CHECK(array_equal(t.sum(0), {{21, 24, 27}, {30, 33, 36}}));
  CHECK(array_equal(t.sum(1), {{5, 7, 9}, {17, 19, 21}, {29, 31, 33}}));
  CHECK(array_equal(t.sum(2), {{6, 15}, {24, 33}, {42, 51}}));
  CHECK(is_close(array<float>{1.1, 2.2}.sum(), 3.3));
  CHECK(is_close(array<int>{1, 2}.sum(), 3l));

  CHECK(v.mean() == 3.5);
  CHECK(t.mean() == 9.5);

  CHECK(array_equal(t.mean(0), array<float>{{7, 8, 9}, {10, 11, 12}}));
  CHECK(array_equal(
      t.mean(1),
      array<float>{{2.5, 3.5, 4.5}, {8.5, 9.5, 10.5}, {14.5, 15.5, 16.5}}));
  CHECK(array_equal(t.mean(2), array<float>{{2, 5}, {8, 11}, {14, 17}}));
}

TEST_CASE("array: softmax") {
  auto v = array<int>{1, 2, 3, 4, 5, 6};
  auto m = array<int>{{7, 8, 9}, {10, 11, 12}};

  auto vsm = v.softmax();
  auto msm = m.softmax();

  CHECK(vsm.sum() == 1);
  CHECK(vsm.all([](auto x) { return x <= 1; }));

  CHECK(array_equal(msm.sum(1), array<float>{1, 1}));
  CHECK(msm.all([](auto x) { return x <= 1; }));
}

TEST_CASE("array: iterators") {
  auto t = array<int>{
      {{1, 2, 3}, {4, 5, 6}},
      {{7, 8, 9}, {10, 11, 12}},
      {{13, 14, 15}, {16, 17, 18}},
  };

  for (auto row : t) {
    for (auto &x : row.elements()) {
      x += 100;
    }
  }

  const auto ct = t;

  int cur = 101;
  for (auto row : ct.rows()) {
    for (const auto &x : row.elements()) {
      CHECK(x == cur++);
    }
  }

  cur = 101;
  for (auto row : ct.rows()) {
    for (auto [a, b, c] : row.rows<3>()) {
      CHECK(a == cur++);
      CHECK(b == cur++);
      CHECK(c == cur++);
    }
  }
}
