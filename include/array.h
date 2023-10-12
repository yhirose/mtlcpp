#pragma once

#include <metal.h>

#include <concepts>
#include <iostream>  // debug...
#include <iterator>
#include <numeric>
#include <ranges>

namespace mtl {

template <typename T>
concept value_type = std::is_same_v<T, float> || std::is_same_v<T, int> ||
                     std::is_same_v<T, unsigned int>;

using shape_type = std::vector<size_t>;

template <value_type T>
class array {
 public:
  array(array &&rhs) = default;
  array(const array &rhs) = default;             // TODO: use GPU
  array &operator=(const array &rhs) = default;  // TODO: use GPU
                                                 //
  array(const shape_type &shape) : shape_(shape) { allocate_buffer_(); }

  //----------------------------------------------------------------------------

  array clone() const {
    array tmp(shape_);
    // TODO: use GPU
    memcpy(tmp.buf_->contents(), buf_->contents(), buf_->length());
    return tmp;
  }

  //----------------------------------------------------------------------------

  bool operator==(const array &rhs) const {
    if (this != &rhs) {
      if (shape_ != rhs.shape_) {
        return false;
      }
      // TODO: use GPU
      return memcmp(buf_->contents(), rhs.buf_->contents(), buf_->length()) ==
             0;
    }
    return true;
  }

  bool operator!=(const array &rhs) const { return !this->operator==(rhs); }

  //----------------------------------------------------------------------------

  size_t length() const { return buf_->length() / sizeof(T); }

  //----------------------------------------------------------------------------

  const shape_type &shape() const { return shape_; }

  size_t shape(size_t i) const {
    // TODO: bounds check
    return shape_[i];
  }

  size_t dimension() const { return shape_.size(); }

  void reshape(const shape_type &shape) {
    // TODO: bounds check
    shape_ = shape;
  }

  //----------------------------------------------------------------------------

  T *data() { return static_cast<T *>(buf_->contents()); }

  const T *data() const { return static_cast<const T *>(buf_->contents()); }

  //----------------------------------------------------------------------------

  T operator()() const { return *data(); }

  T &operator()() { return *data(); }

  T operator[](size_t i) const {
    bounds_check_(i);
    return data()[i];
  }

  T &operator[](size_t i) {
    bounds_check_(i);
    return data()[i];
  }

  T operator()(size_t row, size_t col) const {
    // TODO: bounds check
    // bounds_check_(i);
    return data()[shape_[1] * row + col];
  }

  T &operator()(size_t row, size_t col) {
    // TODO: bounds check
    // bounds_check_(i);
    return data()[shape_[1] * row + col];
  }

  //----------------------------------------------------------------------------

  using iterator = T *;
  iterator begin() { return data(); }
  iterator end() { return data() + length(); }

  using const_iterator = const T *;
  const_iterator cbegin() const { return data(); }
  const_iterator cend() const { return data() + length(); }

  //----------------------------------------------------------------------------

  void set(std::ranges::input_range auto &&r) { std::ranges::copy(r, data()); }
  void set(std::initializer_list<T> l) { std::ranges::copy(l, data()); }

  //----------------------------------------------------------------------------

  void constants(T val) {
    // TODO: use GPU
    for (auto &x : *this) {
      x = val;
    }
  }

  void zeros() { constants(0); };

  void ones() { constants(1); };

  void random() {
    // TODO: use GPU
    for (auto &x : *this) {
      x = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
    }
  }

  //----------------------------------------------------------------------------

  array operator+(const array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_ADD_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_ADD_I);
    }
  }

  array operator-(const array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_SUB_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_SUB_I);
    }
  }

  array operator*(const array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_MUL_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_MUL_I);
    }
  }

  array operator/(const array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_DIV_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_DIV_I);
    }
  }

  array dot(const array &rhs) const {
    // TODO: use GPU
    if (dimension() == 1 && rhs.dimension() == 1 && shape(0) == rhs.shape(0)) {
      array<T> tmp({});

      T val = 0;
      for (size_t i = 0; i < shape(0); i++) {
        val += (*this)[i] * rhs[i];
      }
      tmp() = val;
      return tmp;
    }

    if (dimension() == 2 && rhs.dimension() == 2 && shape(1) == rhs.shape(0)) {
      auto rows = shape(0);
      auto cols = rhs.shape(1);
      array<T> tmp({rows, cols});

      for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
          T val = 0;
          for (size_t i = 0; i < shape(1); i++) {
            val += (*this)(row, i) * rhs(i, col);
          }
          tmp(row, col) = val;
        }
      }
      return tmp;
    }

    if (dimension() == 1 && rhs.dimension() == 2 && shape(0) == rhs.shape(0)) {
      auto rows = 1;
      auto cols = rhs.shape(1);
      array<T> tmp({cols});

      for (size_t col = 0; col < cols; col++) {
        T val = 0;
        for (size_t i = 0; i < shape(0); i++) {
          val += (*this)[i] * rhs(i, col);
        }
        tmp[col] = val;
      }
      return tmp;
    }

    if (dimension() == 2 && rhs.dimension() == 1 && shape(1) == rhs.shape(0)) {
      auto rows = shape(0);
      array<T> tmp({rows});

      for (size_t row = 0; row < rows; row++) {
        T val = 0;
        for (size_t i = 0; i < shape(1); i++) {
          val += (*this)(row, i) * rhs[i];
        }
        tmp[row] = val;
      }
      return tmp;
    }

    throw std::runtime_error("array: can't do `dot` operation.");
  }

  //----------------------------------------------------------------------------

  array operator+(T val) const {
    // TODO: use GPU
    array<T> tmp{*this};
    for (auto &x : tmp) {
      x += val;
    }
    return tmp;
  }

  array operator-(T val) const {
    // TODO: use GPU
    array<T> tmp{*this};
    for (auto &x : tmp) {
      x -= val;
    }
    return tmp;
  }

  array operator*(T val) const {
    // TODO: use GPU
    array<T> tmp{*this};
    for (auto &x : tmp) {
      x *= val;
    }
    return tmp;
  }

  array operator/(T val) const {
    // TODO: use GPU
    array<T> tmp{*this};
    for (auto &x : tmp) {
      x /= val;
    }
    return tmp;
  }

  //----------------------------------------------------------------------------

  auto sum() const { return std::accumulate(cbegin(), cend(), 0); }

  template <typename U = T>
  auto mean() const {
    return std::accumulate(cbegin(), cend(), 0) / static_cast<U>(length());
  }

 private:
  void allocate_buffer_() {
    size_t length = 1;
    for (auto n : shape_) {
      length *= n;
    }
    buf_ = mtl::newBuffer(sizeof(T) * length);
  }

  auto computer_(const array &rhs, mtl::ComputeType id) const {
    if (shape() != rhs.shape()) {
      throw std::runtime_error("array: Invalid operation.");
    }

    array tmp(shape());
    mtl::compute(buf_.get(), rhs.buf_.get(), tmp.buf_.get(), id, sizeof(T));
    return tmp;
  }

  void bounds_check_(size_t i) const {
    if (i >= length()) {
      throw std::runtime_error("array: Index is out of bounds.");
    }
  }

  //----------------------------------------------------------------------------

  shape_type shape_;
  mtl::managed_ptr<MTL::Buffer> buf_;
};

//------------------------------------------------------------------------------

template <typename T>
inline size_t print(std::ostream &os, const array<T> &arr, size_t dim,
                    size_t arr_index) {
  if (dim + 1 == arr.dimension()) {
    size_t i = 0;
    for (; i < arr.shape(dim); i++, arr_index++) {
      if (i > 0) {
        os << ' ';
      }
      os << arr[arr_index];
    }
    return arr_index;
  }

  for (size_t i = 0; i < arr.shape(dim); i++) {
    if (dim == 0 && i > 0) {
      os << "\n ";
    }
    os << '[';
    arr_index = print(os, arr, dim + 1, arr_index);
    os << ']';
  }
  return arr_index;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const array<T> &arr) {
  if (arr.dimension() == 0) {
    os << arr();
  } else {
    os << "[";
    print(os, arr, 0, 0);
    os << "]";
  }
  return os;
}

//------------------------------------------------------------------------------

template <typename T>
inline auto scalar() {
  return array<T>({});
}

template <typename T>
inline auto scalar(T val) {
  auto tmp = array<T>({});
  tmp() = val;
  return tmp;
}

//------------------------------------------------------------------------------

template <typename T>
inline auto vector(size_t length) {
  return array<T>({length});
}

template <typename T>
inline auto vector(std::initializer_list<T> l) {
  auto tmp = vector<T>(l.size());
  tmp.set(l);
  return tmp;
}

template <typename T>
inline auto vector(size_t length, std::ranges::input_range auto &&r) {
  auto tmp = vector<T>(length);
  tmp.set(r);
  return tmp;
}

template <typename T>
inline auto random(size_t length) {
  auto tmp = vector<T>(length);
  tmp.random();
  return tmp;
}

template <typename T>
inline auto constants(size_t length, T val) {
  auto tmp = vector<T>(length);
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
inline auto matrix(size_t row, size_t col) {
  return array<T>({row, col});
}

template <typename T>
inline auto matrix(size_t row, size_t col, std::ranges::input_range auto &&r) {
  auto tmp = matrix<T>(row, col);
  tmp.set(r);
  return tmp;
}

template <typename T>
inline auto random(size_t row, size_t col) {
  auto tmp = matrix<T>(row, col);
  tmp.random();
  return tmp;
}

template <typename T>
inline auto constants(size_t row, size_t col, T val) {
  auto tmp = matrix<T>(row, col);
  tmp.constants(val);
  return tmp;
}

template <typename T>
inline auto zeros(size_t row, size_t col) {
  auto tmp = matrix<T>(row, col);
  tmp.constants(0);
  return tmp;
}

template <typename T>
inline auto ones(size_t row, size_t col) {
  auto tmp = matrix<T>(row, col);
  tmp.constants(1);
  return tmp;
}

};  // namespace mtl
