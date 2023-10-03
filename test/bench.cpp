#include <array.h>
// #include <gpu.h>
// #include <metal.h>

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

using namespace ankerl::nanobench;
using namespace mtlcpp;

template <class T>
void array_add_f_cpu(const T *a, const T *b, T *c, size_t length) {
  for (size_t i = 0; i < length; i++) {
    c[i] = a[i] - b[i];
  }
}

void bench_comparison_between_gpu_and_cpu() {
  size_t epochs = 10;
  const size_t length = 60 * 180 * 10000;

  auto a = GPU::allocate<float>(length);
  auto b = GPU::allocate<float>(length);
  auto out = GPU::allocate<float>(length);

  Bench().epochs(epochs).run("GPU add", [&] {
    GPU::compute(a, b, out, ComputeType::ARRAY_ADD_F, sizeof(float));
  });

  Bench().epochs(epochs).run("CPU add", [&] {
    array_add_f_cpu(a.data<float>(), b.data<float>(), out.data<float>(),
                    length);
  });
}

void bench_array_operations() {
  size_t epochs = 10;
  const size_t length = 60 * 180 * 10000;

  auto a = random<float>(length);
  auto b = random<float>(length);
  auto out = random<float>(length);

  Bench().epochs(epochs).run("ones", [&] {
      ones<float>(length);
  });

  Bench().epochs(epochs).run("random", [&] {
      random<float>(length);
  });

  Bench().epochs(epochs).run("a + b", [&] {
      a + b;
  });
}

int main(void) {
  bench_comparison_between_gpu_and_cpu();
  bench_array_operations();
}
