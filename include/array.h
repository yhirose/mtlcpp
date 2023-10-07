#pragma once

#include <gpu.h>

#include <iterator>
#include <ranges>
#include <sstream>
#include <iostream>

namespace mtlcpp {

template <typename T, typename Backend = GPU>
class Array {
 private:
  size_t length_;
  Backend::Memory buf_;

 public:
  Array(const Array &rhs) = default;
  Array(Array &&rhs) = default;
  Array &operator=(const Array &rhs) = default;

  // This constructor is needed to avoid `std::ranges::input_range`
  Array(Array &rhs) {
    length_ = rhs.length_;
    buf_ = rhs.buf_;
  }

  Array(size_t length) : length_(length) {
    buf_ = Backend::allocate(buf_length());
  }

  Array(std::initializer_list<T> l) : length_(l.size()) {
    buf_ = Backend::allocate(buf_length());
    std::ranges::copy(l, data());
  }

  Array(std::ranges::input_range auto &&r) : length_(std::ranges::distance(r)) {
    buf_ = Backend::allocate(buf_length());
    std::ranges::copy(r, data());
  }

  Array copy() const {
    Array tmp(length_);
    memcpy(tmp.buf_.data(), buf_.data(), buf_.length());
    return tmp;
  }

  bool operator==(const Array &rhs) const {
    if (this != &rhs) {
      if (length() != rhs.length()) {
        return false;
      }
      return memcmp(buf_.data(), rhs.buf_.data(), buf_.length()) == 0;
    }
    return true;
  }

  bool operator!=(const Array &rhs) const { return !this->operator==(rhs); }

  size_t length() const { return length_; }

  T *data() { return static_cast<T *>(buf_.data()); }

  const T *data() const { return static_cast<const T *>(buf_.data()); }

  T operator[](size_t i) const {
    // TODO: range check...
    return data()[i];
  }

  T &operator[](size_t i) {
    // TODO: check range...
    return data()[i];
  }

  using iterator = T *;
  iterator begin() { return data(); }
  iterator end() { return data() + length_; }

  Array operator+(const Array &rhs) const {
    return compute(rhs, ComputeType::ARRAY_ADD_F);
  }

  Array operator-(const Array &rhs) const {
    return compute(rhs, ComputeType::ARRAY_SUB_F);
  }

  Array operator*(const Array &rhs) const {
    return compute(rhs, ComputeType::ARRAY_MUL_F);
  }

  Array operator/(const Array &rhs) const {
    return compute(rhs, ComputeType::ARRAY_DIV_F);
  }

  void constants(T val) {
    for (auto &x : *this) {
      x = val;
    }
  }

  void zeros() { constants(0); };

  void ones() { constants(1); };

  void random() {
    for (auto &x : *this) {
      x = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }
  }

 private:
  size_t buf_length() const { return sizeof(T) * length_; }

  Array compute(const Array &rhs, ComputeType id) const {
    if (length() != rhs.length()) {
      std::stringstream ss;
      ss << "array: Invalid operation.";
      throw std::runtime_error(ss.str());
    }

    Array tmp(length_);
    Backend::compute(buf_, rhs.buf_, tmp.buf_, id, sizeof(T));
    return tmp;
  }
};

//-----------------------------------------------------------------------------

template <typename T>
inline auto random(size_t length) {
  Array<T> arr(length);
  arr.random();
  return arr;
}

template <typename T>
inline auto constants(size_t length, T val) {
  Array<T> tmp(length);
  tmp.constants(val);
  return tmp;
}

template <typename T>
inline auto zeros(size_t length) {
  return constants<T>(length, 0);
}

template <typename T>
inline auto ones(size_t length) {
  return constants<T>(length, 1);
}

//-----------------------------------------------------------------------------

template <typename T>
std::ostream &operator<<(std::ostream &os, const Array<T> &arr) {
  os << "mp::Array([";
  for (size_t i = 0; i < arr.length(); i++) {
    if (i > 0) {
      os << ' ';
    }
    os << arr[i];
  }
  os << "])";
  return os;
}

};  // namespace mtlcpp

