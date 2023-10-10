#pragma once

#include <metal.h>

#include <concepts>
#include <iterator>
#include <numeric>
#include <ranges>
#include <sstream>

namespace mtlcpp {

template <typename T>
concept ElementType = std::is_same_v<T, float> || std::is_same_v<T, int> ||
                      std::is_same_v<T, unsigned int>;

template <ElementType T>
class Array {
 private:
  size_t length_;
  mtl::managed_ptr<MTL::Buffer> buf_;

 public:
  Array(const Array &rhs) = default;
  Array(Array &&rhs) = default;
  Array &operator=(const Array &rhs) = default;

  // This constructor is to avoid incorrect `std::ranges::input_range` usage
  Array(Array &rhs) {
    length_ = rhs.length_;
    buf_ = rhs.buf_;
  }

  bool operator==(const Array &rhs) const {
    if (this != &rhs) {
      if (length() != rhs.length()) {
        return false;
      }
      return memcmp(buf_->contents(), rhs.buf_->contents(), buf_->length()) ==
             0;
    }
    return true;
  }

  bool operator!=(const Array &rhs) const { return !this->operator==(rhs); }

  //----------------------------------------------------------------------------

  Array(size_t length) : length_(length) {
    buf_ = mtl::newBuffer(buf_length());
  }

  Array(std::initializer_list<T> l) : length_(l.size()) {
    buf_ = mtl::newBuffer(buf_length());
    std::ranges::copy(l, data());
  }

  Array(std::ranges::input_range auto &&r) : length_(std::ranges::distance(r)) {
    buf_ = mtl::newBuffer(buf_length());
    std::ranges::copy(r, data());
  }

  //----------------------------------------------------------------------------

  Array copy() const {
    Array tmp(length_);
    memcpy(tmp.buf_->contents(), buf_->contents(), buf_->length());
    return tmp;
  }

  //----------------------------------------------------------------------------

  size_t length() const { return length_; }

  //----------------------------------------------------------------------------

  T *data() { return static_cast<T *>(buf_->contents()); }

  const T *data() const { return static_cast<const T *>(buf_->contents()); }

  //----------------------------------------------------------------------------

  T operator[](size_t i) const {
    bounds_check(i);
    return data()[i];
  }

  T &operator[](size_t i) {
    bounds_check(i);
    return data()[i];
  }

  //----------------------------------------------------------------------------

  using iterator = T *;
  iterator begin() { return data(); }
  iterator end() { return data() + length_; }

  using const_iterator = const T *;
  const_iterator cbegin() const { return data(); }
  const_iterator cend() const { return data() + length_; }

  //----------------------------------------------------------------------------

  void constants(T val) {
    for (auto &x : *this) {
      x = val;
    }
  }

  void zeros() { constants(0); };

  void ones() { constants(1); };

  void random(size_t times = 1, T bias = 0) {
    for (auto &x : *this) {
      x = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) *
              times +
          bias;
    }
  }

  //----------------------------------------------------------------------------

  Array operator+(const Array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return compute(rhs, mtl::ComputeType::ARRAY_ADD_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return compute(rhs, mtl::ComputeType::ARRAY_ADD_I);
    } else {
      return compute(rhs, mtl::ComputeType::ARRAY_ADD_U);
    }
  }

  Array operator-(const Array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return compute(rhs, mtl::ComputeType::ARRAY_SUB_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return compute(rhs, mtl::ComputeType::ARRAY_SUB_I);
    } else {
      return compute(rhs, mtl::ComputeType::ARRAY_SUB_U);
    }
  }

  Array operator*(const Array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return compute(rhs, mtl::ComputeType::ARRAY_MUL_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return compute(rhs, mtl::ComputeType::ARRAY_MUL_I);
    } else {
      return compute(rhs, mtl::ComputeType::ARRAY_MUL_U);
    }
  }

  Array operator/(const Array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return compute(rhs, mtl::ComputeType::ARRAY_DIV_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return compute(rhs, mtl::ComputeType::ARRAY_DIV_I);
    } else {
      return compute(rhs, mtl::ComputeType::ARRAY_DIV_U);
    }
  }

  //----------------------------------------------------------------------------

  auto sum() const { return std::accumulate(cbegin(), cend(), 0); }

  template <typename U = T>
  auto mean() const {
    return std::accumulate(cbegin(), cend(), 0) / static_cast<U>(length());
  }

 private:
  auto buf_length() const { return sizeof(T) * length_; }

  auto compute(const Array &rhs, mtl::ComputeType id) const {
    if (length() != rhs.length()) {
      std::stringstream ss;
      ss << "array: Invalid operation.";
      throw std::runtime_error(ss.str());
    }

    Array tmp(length_);
    mtl::compute(buf_.get(), rhs.buf_.get(), tmp.buf_.get(), id, sizeof(T));
    return tmp;
  }

  void bounds_check(size_t i) const {
    if (i >= length_) {
      std::stringstream ss;
      ss << "array: Index is out of bounds.";
      throw std::runtime_error(ss.str());
    }
  }
};

//------------------------------------------------------------------------------

template <typename T>
inline auto random(size_t length, size_t times = 1, T bias = 0) {
  Array<T> arr(length);
  arr.random(times, bias);
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

//------------------------------------------------------------------------------

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

