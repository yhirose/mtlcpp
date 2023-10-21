#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define ANKERL_NANOBENCH_IMPLEMENT

#include <array.h>

#include <eigen3/Eigen/Core>

#include "doctest.h"
#include "nanobench.h"

using namespace ankerl::nanobench;

TEST_CASE("add") {
  size_t epochs = 100;
  const size_t n = 10'000'000;

  auto a = mtl::ones<float>(n);
  auto b = mtl::ones<float>(n);
  auto e = mtl::constants<float>(n, 2);

  auto c = mtl::vector<float>(n);

  mtl::device = mtl::Device::GPU;
  Bench().minEpochIterations(epochs).run("GPU: a + b", [&] { c = a + b; });
  CHECK(mtl::array_equal(e, c));

  auto aa = Eigen::Vector<float, Eigen::Dynamic>::Ones(n);
  auto bb = Eigen::Vector<float, Eigen::Dynamic>::Ones(n);
  auto ee = Eigen::Vector<float, Eigen::Dynamic>::Constant(n, 2);

  auto cc = Eigen::Vector<float, Eigen::Dynamic>(n);

  Bench().minEpochIterations(epochs).run("Eigen: a + b", [&] { cc = aa + bb; });
  CHECK(ee == cc);
}

TEST_CASE("dot") {
  auto a = mtl::constants<float>(1000, 100, 1);
  auto b = mtl::constants<float>(100, 10, 1);
  auto e = mtl::constants<float>(1000, 10, 100);

  auto c = mtl::matrix<float>(1000, 10);

  mtl::device = mtl::Device::CPU;
  Bench().run("CPU: a.dot(b)", [&] { c = a.dot(b); });
  CHECK(mtl::array_equal(e, c));

  auto aa =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Ones(1000, 100);
  auto bb = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Ones(100, 10);
  auto ee = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      1000, 10, 100);

  auto cc = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(1000, 10);

  Bench().run("Eigen: a * b", [&] { cc = aa * bb; });
  CHECK(ee == cc);
}
