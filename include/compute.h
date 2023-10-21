#pragma once

#include <numeric>

namespace mtl {

template <typename T>
concept value_type =
    std::same_as<T, float> || std::same_as<T, int> || std::same_as<T, bool>;

enum class Device {
  GPU = 0,
  CPU,
};

enum class Operation {
  Add = 0,
  Sub,
  Mul,
  Div,
};

enum class DataType {
  Float = 0,
  Integer,
};

static Device device = Device::GPU;

};  // namespace mtl
