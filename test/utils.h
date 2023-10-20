#pragma once

#include <cstdlib>
#include <ranges>

template <typename T, typename U>
inline bool verify_array(const T* A, const T* B, const T* OUT, size_t length,
                         U fn) {
  size_t err = 0;
  for (size_t i = 0; i < length; i++) {
    if (OUT[i] != fn(A[i], B[i])) {
      return false;
    }
  }
  return true;
}

template <typename T, typename U>
inline bool verify_array_tolerant(const T* A, const T* B, const T* OUT,
                                  size_t length, U fn) {
  size_t err = 0;
  for (size_t i = 0; i < length; i++) {
    if (std::abs(OUT[i] - fn(A[i], B[i])) > 1e-3) {
      err++;
    }
  }
  if (err == 0) {
    return true;
  } else {
    auto ratio = static_cast<double>(err) / length * 100.0;
    return ratio < 0.001;
  }
}

inline auto itoa(size_t size) {
  return std::views::iota(1) | std::views::take(size);
}
