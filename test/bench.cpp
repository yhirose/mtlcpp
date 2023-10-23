#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define ANKERL_NANOBENCH_IMPLEMENT

#include <array.h>

#include <eigen3/Eigen/Core>
#include <xtensor.hpp>

#include "doctest.h"
#include "nanobench.h"

using namespace ankerl::nanobench;

TEST_CASE("add") {
  const size_t n = 10'000'000;

  auto a = mtl::ones<float>({n});
  auto b = mtl::ones<float>({n});
  auto e = mtl::array<float>({n}, 2);
  auto c = mtl::array<float>();
  Bench().run("mtlcpp: a + b", [&] { c = a + b; });
  CHECK(mtl::array_equal(e, c));

  auto aa = Eigen::Vector<float, Eigen::Dynamic>::Ones(n);
  auto bb = Eigen::Vector<float, Eigen::Dynamic>::Ones(n);
  auto ee = Eigen::Vector<float, Eigen::Dynamic>::Constant(n, 2);
  auto cc = Eigen::Vector<float, Eigen::Dynamic>(n);
  Bench().run("Eigen: a + b", [&] { cc = aa + bb; });
  CHECK(ee == cc);

  auto aaa = xt::ones<float>({n});
  auto bbb = xt::ones<float>({n});
  auto eee = 2.0 * xt::ones<float>({n});
  auto ccc = xt::xarray<float>({n});
  Bench().run("xtensor: a + b", [&] { ccc = aaa + bbb; });
  CHECK(eee == ccc);
}

TEST_CASE("dot") {
  auto a = mtl::ones<float>({1000, 100});
  auto b = mtl::ones<float>({100, 10});
  auto e = mtl::array<float>({1000, 10}, 100);
  auto c = mtl::array<float>();
  Bench().run("mtlcpp: a.dot(b)", [&] { c = a.dot(b); });
  CHECK(mtl::array_equal(e, c));

  auto aa =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Ones(1000, 100);
  auto bb = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Ones(100, 10);
  auto ee = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      1000, 10, 100);
  auto cc = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>();
  Bench().run("Eigen: a * b", [&] { cc = aa * bb; });
  CHECK(ee == cc);
}
