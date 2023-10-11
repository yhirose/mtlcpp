#pragma once

#include <metal.h>

#include <concepts>
#include <iterator>
#include <numeric>
#include <ranges>
#include <sstream>

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

  size_t length() const {
    // TODO: cache length
    size_t l = 1;
    for (auto n : shape_) {
      l *= n;
    }
    return l;
  }

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

  void copy(std::ranges::input_range auto &&r) { std::ranges::copy(r, data()); }

  //----------------------------------------------------------------------------

  void constants(T val) {
    // TODO: use GPU
    for (auto &x : *this) {
      x = val;
    }
  }

  void zeros() { constants(0); };

  void ones() { constants(1); };

  void random(size_t times, T bias) {
    // TODO: use GPU
    for (auto &x : *this) {
      x = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) *
              times +
          bias;
    }
  }

  //----------------------------------------------------------------------------

  array operator+(const array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_ADD_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_ADD_I);
    } else {
      return computer_(rhs, mtl::ComputeType::ARRAY_ADD_U);
    }
  }

  array operator-(const array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_SUB_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_SUB_I);
    } else {
      return computer_(rhs, mtl::ComputeType::ARRAY_SUB_U);
    }
  }

  array operator*(const array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_MUL_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_MUL_I);
    } else {
      return computer_(rhs, mtl::ComputeType::ARRAY_MUL_U);
    }
  }

  array operator/(const array &rhs) const {
    if constexpr (std::is_same_v<T, float>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_DIV_F);
    } else if constexpr (std::is_same_v<T, int>) {
      return computer_(rhs, mtl::ComputeType::ARRAY_DIV_I);
    } else {
      return computer_(rhs, mtl::ComputeType::ARRAY_DIV_U);
    }
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
  void allocate_buffer_() { buf_ = mtl::newBuffer(sizeof(T) * length()); }

  auto computer_(const array &rhs, mtl::ComputeType id) const {
    if (shape() != rhs.shape()) {
      std::stringstream ss;
      ss << "array: Invalid operation.";
      throw std::runtime_error(ss.str());
    }

    array tmp(shape());
    mtl::compute(buf_.get(), rhs.buf_.get(), tmp.buf_.get(), id, sizeof(T));
    return tmp;
  }

  void bounds_check_(size_t i) const {
    if (i >= length()) {
      std::stringstream ss;
      ss << "array: Index is out of bounds.";
      throw std::runtime_error(ss.str());
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
  os << "[";
  print(os, arr, 0, 0);
  os << "]";
  return os;
}

//------------------------------------------------------------------------------

namespace vec {

template <typename T>
inline auto vector(size_t length) {
  return array<T>({length});
}

template <typename T>
inline auto vector(std::initializer_list<T> l) {
  array<T> tmp({l.size()});
  tmp.copy(l);
  return tmp;
}

template <typename T>
inline auto vector(size_t length, std::ranges::input_range auto &&r) {
  array<T> tmp({length});
  tmp.copy(r);
  return tmp;
}

template <typename T>
inline auto random(size_t length, size_t times = 1, size_t bias = 0) {
  auto tmp = vector<T>(length);
  tmp.random(times, bias);
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

};  // namespace vec

//------------------------------------------------------------------------------

namespace mat {

template <typename T>
inline auto matrix(size_t row, size_t col) {
  return array<T>({row, col});
}

template <typename T>
inline auto matrix(size_t row, size_t col, std::ranges::input_range auto &&r) {
  auto tmp = matrix<T>(row, col);
  tmp.copy(r);
  return tmp;
}

template <typename T>
inline auto random(size_t row, size_t col, size_t times = 1, size_t bias = 0) {
  auto tmp = matrix<T>(row, col);
  tmp.random(times, bias);
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

};  // namespace mat

};  // namespace mtl
