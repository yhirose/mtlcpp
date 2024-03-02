#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define ANKERL_NANOBENCH_IMPLEMENT

#include <array.h>

#include <eigen3/Eigen/Core>

#include "nanobench.h"

using namespace ankerl::nanobench;

void add() {
  const size_t n = 10'000'000;

  auto a = mtl::ones<float>({n});
  auto b = mtl::ones<float>({n});
  auto e = mtl::array<float>({n}, 2);
  auto c = mtl::array<float>();

  mtl::device = mtl::Device::CPU;
  Bench().run("CPU: a + b", [&] { c = a + b; });

  mtl::device = mtl::Device::GPU;
  Bench().minEpochIterations(100).run("GPU: a + b", [&] { c = a + b; });

  auto aa = Eigen::Vector<float, Eigen::Dynamic>::Ones(n);
  auto bb = Eigen::Vector<float, Eigen::Dynamic>::Ones(n);
  auto cc = Eigen::Vector<float, Eigen::Dynamic>(n);

  Bench().minEpochIterations(100).run("Eigen: a + b", [&] { cc = aa + bb; });
}

void dot() {
  auto a = mtl::ones<float>({1000, 1000});
  auto b = mtl::ones<float>({1000, 100});
  auto e = mtl::array<float>({1000, 100}, 1000);
  auto c = mtl::array<float>();

  mtl::device = mtl::Device::CPU;
  Bench().run("CPU: a.dot(b)", [&] { c = a.dot(b); });

  mtl::device = mtl::Device::GPU;
  Bench().minEpochIterations(100).run("GPU: a.dot(b)", [&] { c = a.dot(b); });

  auto aa =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Ones(1000, 1000);
  auto bb =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Ones(1000, 100);
  auto cc = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>();

  Bench().minEpochIterations(100).run("Eigen: a * b", [&] { cc = aa * bb; });
}

int main(void) {
  add();
  dot();
}
