#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <array.h>

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

using namespace ankerl::nanobench;
using namespace mtl;

template <class T>
void array_add_f_cpu(const T *a, const T *b, T *c, size_t length) {
  for (size_t i = 0; i < length; i++) {
    c[i] = a[i] + b[i];
  }
}

void bench_comparison_between_gpu_and_cpu() {
  size_t epochs = 10;
  const size_t length = 100'000'000;

  auto a = mtl::newBuffer(sizeof(float) * length);
  auto b = mtl::newBuffer(sizeof(float) * length);
  auto out = mtl::newBuffer(sizeof(float) * length);

  Bench().epochs(epochs).run("GPU add", [&] {
    mtl::compute(a.get(), b.get(), out.get(), mtl::ComputeType::ARRAY_ADD_F,
                 sizeof(float));
  });

  Bench().epochs(epochs).run("CPU add", [&] {
    array_add_f_cpu<float>(static_cast<float *>(a->contents()),
                           static_cast<float *>(b->contents()),
                           static_cast<float *>(out->contents()), length);
  });
}

void bench_array_operations() {
  size_t epochs = 10;
  const size_t length = 100'000'000;

  auto a = vec::random<float>(length);
  auto b = vec::random<float>(length);
  auto out = vec::random<float>(length);

  Bench().epochs(epochs).run("ones", [&] { vec::ones<float>(length); });
  Bench().epochs(epochs).run("random", [&] { vec::random<float>(length); });
  Bench().epochs(epochs).run("a + b", [&] { a + b; });
}

int main(void) {
  bench_comparison_between_gpu_and_cpu();
  bench_array_operations();
}
