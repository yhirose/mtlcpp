#pragma once

#include <numeric>

namespace mtl {

template <typename T>
concept value_type =
    std::same_as<T, float> || std::same_as<T, int> || std::same_as<T, bool>;

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

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

};  // namespace mtl
