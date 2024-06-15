#pragma once

#include <metal.h>

#include <concepts>
#include <iostream>
#include <iterator>
#include <limits>
#include <ranges>

namespace mtl {

using shape_type = std::vector<size_t>;
using strides_type = shape_type;

//------------------------------------------------------------------------------

template <value_type T>
class array {
 public:
  array() = default;
  array(array &&rhs) = default;
  array(const array &rhs) = default;
  array &operator=(const array &rhs) = default;

  array(const shape_type &shape, T val);
  array(const shape_type &shape, std::input_iterator auto it);
  array(const shape_type &shape, std::ranges::input_range auto &&r);
  array(std::ranges::input_range auto &&r);
  array(T val);

  array(nested_initializer_list<T, 1> l);
  array(nested_initializer_list<T, 2> l);
  array(nested_initializer_list<T, 3> l);
  array(nested_initializer_list<T, 4> l);

  //----------------------------------------------------------------------------

  template <value_type U = T>
  array<U> clone() const;

  //----------------------------------------------------------------------------

  array<bool> operator==(const array &rhs) const;
  array<bool> operator!=(const array &rhs) const;
  array<bool> operator>(const array &rhs) const;
  array<bool> operator<(const array &rhs) const;
  array<bool> operator>=(const array &rhs) const;
  array<bool> operator<=(const array &rhs) const;

  //----------------------------------------------------------------------------

  size_t buffer_element_count() const;
  size_t buffer_bytes() const;

  T *buffer_data();
  const T *buffer_data() const;

  //----------------------------------------------------------------------------

  size_t element_count() const;
  size_t length() const;

  size_t dimension() const;
  const shape_type &shape() const;
  const strides_type &strides() const;

  void reshape(const shape_type &shape);

  // TODO: can return a reference for performance?
  const auto broadcast(const shape_type &target_shape) const;

  array transpose() const;

  //----------------------------------------------------------------------------

  T at() const;
  T &at();

  T at(size_t i) const;
  T &at(size_t i);

  T at(size_t x, size_t y) const;
  T &at(size_t x, size_t y);

  T at(size_t x, size_t y, size_t z) const;
  T &at(size_t x, size_t y, size_t z);

  T at(const std::vector<size_t> &position) const;
  T &at(const std::vector<size_t> &position);

  template <size_t I>
  auto take() const;

  //----------------------------------------------------------------------------

  array operator[](size_t row) const;

  //----------------------------------------------------------------------------

  auto element_begin();
  auto element_end();

  auto element_cbegin() const;
  auto element_cend() const;

  auto elements();
  auto elements() const;

  //----------------------------------------------------------------------------

  auto begin();
  auto end();

  auto begin() const;
  auto end() const;

  auto cbegin() const;
  auto cend() const;

  template <size_t N = 0>
  auto rows();

  template <size_t N = 0>
  auto rows() const;

  //----------------------------------------------------------------------------

  void set(std::input_iterator auto it);
  void set(std::initializer_list<T> l);

  //----------------------------------------------------------------------------

  void constants(T val);
  void zeros();
  void ones();
  void random();

  //----------------------------------------------------------------------------

  array operator+(const array &rhs) const;
  array operator-(const array &rhs) const;
  array operator*(const array &rhs) const;
  array operator/(const array &rhs) const;

  array pow(const array &rhs) const;

  void operator+=(const array &rhs);
  void operator-=(const array &rhs);
  void operator*=(const array &rhs);
  void operator/=(const array &rhs);

  //----------------------------------------------------------------------------

  array dot(const array &rhs) const;

  array linear(const array& W, const array& b) const;


  //----------------------------------------------------------------------------

  array<float> sigmoid() const;

  //----------------------------------------------------------------------------

  T sum() const;
  array sum(size_t axis) const;

  float mean() const;
  array<float> mean(size_t axis) const;

  T min() const;
  T max() const;

  size_t count() const;

  bool all(arithmetic auto val) const;
  template <typename U>
  bool all(U fn) const;

  array<float> softmax() const;
  auto argmax() const;

  float mean_square_error(const array &rhs) const;

  //----------------------------------------------------------------------------

  std::string print_shape_type(const shape_type &shape) const;
  std::string print_shape() const;
  std::string print_strides() const;
  std::string print_data_type() const;
  std::string print_info() const;
  std::string print_array() const;

 private:
  shape_type shape_;
  strides_type strides_;
  metal::storage storage_;

  //----------------------------------------------------------------------------

  void allocate_buffer_();

  //----------------------------------------------------------------------------

  void copy_initializer_list_(const auto &l);

  //----------------------------------------------------------------------------

  void bounds_check_(size_t i) const;
  void bounds_check_(size_t x, size_t y) const;
  void bounds_check_(size_t x, size_t y, size_t z) const;

  //----------------------------------------------------------------------------

  template <typename U>
  void enumerate_position_(size_t shape_index, std::vector<size_t> &position,
                           U fn) const;
  template <typename U>
  void enumerate_position_(U fn) const;

  //----------------------------------------------------------------------------

  static auto broadcast_(const array &lhs, const array &rhs, auto cb);

  template <value_type U = T>
  array<U> apply_binary_operation_(const array &rhs, auto ope) const;

  //----------------------------------------------------------------------------

  enum class ArithmeticOperation {
    Add = 0,
    Sub,
    Mul,
    Div,
    Pow,
  };

  static auto gpu_arithmetic_operation_(const array &lhs, const array &rhs,
                                        ArithmeticOperation ope);
  static auto cpu_arithmetic_operation_(const array &lhs, const array &rhs,
                                        ArithmeticOperation ope);

  static auto arithmetic_operation_(const array &lhs, const array &rhs,
                                    ArithmeticOperation ope);

  //----------------------------------------------------------------------------

  static array cpu_dot_operation_(const array &lhs, const array &rhs);
  static array gpu_dot_operation_(const array &lhs, const array &rhs);
  template <typename U>
  array dot_operation_(const array &rhs, U fn) const;
};

//----------------------------------------------------------------------------

template <value_type T>
std::ostream &operator<<(std::ostream &os, const array<T> &arr);

//----------------------------------------------------------------------------

template <value_type T>
array<T> operator+(auto lhs, const array<T> &rhs);

template <value_type T>
array<T> operator-(auto lhs, const array<T> &rhs);

template <value_type T>
array<T> operator*(auto lhs, const array<T> &rhs);

template <value_type T>
array<T> operator/(auto lhs, const array<T> &rhs);

//----------------------------------------------------------------------------

template <value_type T, value_type U>
array<T> where(const array<U> &cond, T x, T y);

//----------------------------------------------------------------------------

template <value_type T>
bool array_equal(const array<T> &a, const array<T> &b);

template <std::floating_point T, std::floating_point U>
bool is_close(T a, U b, float tolerance = 1e-3);

template <std::integral T, std::integral U>
bool is_close(T a, U b);

template <value_type T>
bool allclose(const array<T> &a, const array<T> &b, float tolerance = 1e-3);

//----------------------------------------------------------------------------

template <value_type T>
auto empty(const shape_type &shape);

template <value_type T>
auto zeros(const shape_type &shape);

template <value_type T>
auto ones(const shape_type &shape);

auto random(const shape_type &shape);

//-----------------------------------------------------------------------------
// Implementation
//-----------------------------------------------------------------------------

template <value_type T>
inline array<T>::array(const shape_type &shape, T val) {
  reshape(shape);
  allocate_buffer_();
  constants(val);
}

template <value_type T>
inline array<T>::array(const shape_type &shape, std::input_iterator auto it) {
  reshape(shape);
  allocate_buffer_();
  set(it);
}

template <value_type T>
inline array<T>::array(const shape_type &shape,
                       std::ranges::input_range auto &&r) {
  reshape(shape);
  allocate_buffer_();
  set(r.begin());
}

template <value_type T>
inline array<T>::array(std::ranges::input_range auto &&r) {
  size_t element_count = std::ranges::distance(r);
  reshape({element_count});
  allocate_buffer_();
  set(r.begin());
}

template <value_type T>
inline array<T>::array(T val) : array(shape_type({}), T{}) {
  *buffer_data() = val;
}

template <typename T>
struct depth_ {
  static constexpr size_t value = 0;
};

template <typename T>
struct depth_<std::initializer_list<T>> {
  static constexpr size_t value = 1 + depth_<T>::value;
};

template <size_t I>
struct shape_value_ {
  template <typename T>
  static constexpr size_t value(T l) {
    return l.size() == 0 ? 0 : shape_value_<I - 1>::value(*l.begin());
  }
};

template <>
struct shape_value_<0> {
  template <typename T>
  static constexpr size_t value(T l) {
    return l.size();
  }
};

template <typename T, size_t... I>
constexpr shape_type shape_(T l, std::index_sequence<I...>) {
  return {shape_type::value_type(shape_value_<I>::value(l))...};
}

template <typename T>
constexpr size_t nested_initializer_list_dimension_() {
  return depth_<T>::value;
};

template <typename T>
constexpr shape_type nested_initializer_list_shape_(T l) {
  return shape_<T>(
      l, std::make_index_sequence<nested_initializer_list_dimension_<T>()>());
}

template <value_type T>
inline array<T>::array(nested_initializer_list<T, 1> l)
    : array(nested_initializer_list_shape_(l), T{}) {
  copy_initializer_list_(l);
}

template <value_type T>
inline array<T>::array(nested_initializer_list<T, 2> l)
    : array(nested_initializer_list_shape_(l), T{}) {
  copy_initializer_list_(l);
}

template <value_type T>
inline array<T>::array(nested_initializer_list<T, 3> l)
    : array(nested_initializer_list_shape_(l), T{}) {
  copy_initializer_list_(l);
}

template <value_type T>
inline array<T>::array(nested_initializer_list<T, 4> l)
    : array(nested_initializer_list_shape_(l), T{}) {
  copy_initializer_list_(l);
}

//----------------------------------------------------------------------------

template <value_type T>
template <value_type U>
inline array<U> array<T>::clone() const {
  auto tmp = array<U>(shape_, U{});
  for (size_t i = 0; i < element_count(); i++) {
    tmp.at(i) = static_cast<U>(at(i));
  }
  return tmp;
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<bool> array<T>::operator==(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs == rhs; });
}

template <value_type T>
inline array<bool> array<T>::operator!=(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs != rhs; });
}

template <value_type T>
inline array<bool> array<T>::operator>(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs > rhs; });
}

template <value_type T>
inline array<bool> array<T>::operator>=(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs >= rhs; });
}

template <value_type T>
inline array<bool> array<T>::operator<(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs < rhs; });
}

template <value_type T>
inline array<bool> array<T>::operator<=(const array &rhs) const {
  return apply_binary_operation_<bool>(
      rhs, [](auto lhs, auto rhs) { return lhs <= rhs; });
}

//----------------------------------------------------------------------------

template <value_type T>
inline size_t array<T>::buffer_element_count() const {
  return storage_.len;
}

template <value_type T>
inline size_t array<T>::buffer_bytes() const {
  return storage_.len * sizeof(T);
}

template <value_type T>
inline T *array<T>::buffer_data() {
  return static_cast<T *>(storage_.buf->contents()) + storage_.off;
}

template <value_type T>
inline const T *array<T>::buffer_data() const {
  return static_cast<const T *>(storage_.buf->contents()) + storage_.off;
}

//----------------------------------------------------------------------------

template <value_type T>
inline size_t array<T>::element_count() const {
  // TODO: cache
  size_t count = 1;
  for (auto n : shape_) {
    count *= n;
  }
  return count;
}

template <value_type T>
inline size_t array<T>::length() const {
  if (shape_.empty()) {
    throw std::runtime_error("array: cannot call with a scalar value.");
  }
  return shape_[0];
}

template <value_type T>
inline size_t array<T>::dimension() const {
  return shape_.size();
}

template <value_type T>
inline const shape_type &array<T>::shape() const {
  return shape_;
}

template <value_type T>
inline const strides_type &array<T>::strides() const {
  return strides_;
}

template <value_type T>
inline void array<T>::reshape(const shape_type &shape) {
  // TODO: check the shape
  shape_ = shape;

  // strides
  strides_.clear();
  strides_.push_back(1);
  if (!strides_.empty()) {
    for (int i = shape.size() - 1; i > 0; i--) {
      auto n = strides_.front() * shape[i];
      strides_.insert(strides_.begin(), n);
    }
  }
}

template <value_type T>
inline const auto array<T>::broadcast(const shape_type &target_shape) const {
  if (target_shape.size() < dimension()) {
    throw std::runtime_error("array: invalid shape for broadcast.");
  } else if (target_shape.size() == dimension()) {
    return *this;
  }

  auto diff = target_shape.size() - dimension();
  for (size_t i = 0; i < dimension(); i++) {
    if (shape_[i] != target_shape[i + diff]) {
      throw std::runtime_error("array: invalid shape for broadcast.");
    }
  }

  array tmp = *this;
  tmp.shape_ = target_shape;

  // strides
  tmp.strides_.clear();
  tmp.strides_.push_back(1);
  if (!strides_.empty()) {
    for (int i = target_shape.size() - 1; i > 0; i--) {
      auto n = i <= diff ? 0 : tmp.strides_.front() * target_shape[i];
      tmp.strides_.insert(tmp.strides_.begin(), n);
    }
  }
  return tmp;
}

template <value_type T>
inline array<T> array<T>::transpose() const {
  if (dimension() == 1) {
    auto tmp = clone();
    tmp.reshape({1, element_count()});

    auto it = element_cbegin();
    for (size_t col = 0; col < element_count(); col++) {
      tmp.at(0, col) = *it;
      ++it;
    }
    return tmp;
  }

  if (dimension() == 2) {
    if (shape_[0] == 1) {
      auto tmp = clone();
      tmp.reshape({element_count()});

      auto it = element_cbegin();
      for (size_t row = 0; row < element_count(); row++) {
        tmp.at(row) = *it;
        ++it;
      }
      return tmp;
    } else {
      auto shape = shape_;
      std::ranges::reverse(shape);

      auto tmp = clone();
      tmp.reshape(shape);

      auto it = element_cbegin();
      for (size_t col = 0; col < shape[1]; col++) {
        for (size_t row = 0; row < shape[0]; row++) {
          tmp.at(row, col) = *it;
          ++it;
        }
      }
      return tmp;
    }
  }

  if (dimension() == 3) {
    auto shape = shape_;
    std::ranges::reverse(shape);

    auto tmp = clone();
    tmp.reshape(shape);

    auto it = element_cbegin();
    for (size_t z = 0; z < shape[2]; z++) {
      for (size_t y = 0; y < shape[1]; y++) {
        for (size_t x = 0; x < shape[0]; x++) {
          tmp.at(x, y, z) = *it;
          ++it;
        }
      }
    }
    return tmp;
  }

  throw std::runtime_error("array: can't do `transpose` operation.");
}

//----------------------------------------------------------------------------

template <value_type T>
inline T array<T>::at() const {
  return *buffer_data();
}

template <value_type T>
inline T &array<T>::at() {
  return *buffer_data();
}

template <value_type T>
inline T array<T>::at(size_t i) const {
  bounds_check_(i);
  return buffer_data()[i % buffer_element_count()];
}

template <value_type T>
inline T &array<T>::at(size_t i) {
  bounds_check_(i);
  return buffer_data()[i % buffer_element_count()];
}

template <value_type T>
inline T array<T>::at(size_t x, size_t y) const {
  bounds_check_(x, y);
  return buffer_data()[strides_[0] * x + y];
}

template <value_type T>
inline T &array<T>::at(size_t x, size_t y) {
  bounds_check_(x, y);
  return buffer_data()[strides_[0] * x + y];
}

template <value_type T>
inline T array<T>::at(size_t x, size_t y, size_t z) const {
  bounds_check_(x, y, z);
  return buffer_data()[(strides_[0] * x) + (strides_[1] * y) + z];
}

template <value_type T>
inline T &array<T>::at(size_t x, size_t y, size_t z) {
  bounds_check_(x, y, z);
  return buffer_data()[(strides_[0] * x) + (strides_[1] * y) + z];
}

template <value_type T>
inline T array<T>::at(const std::vector<size_t> &position) const {
  // TODO: bounds_check_(position);
  size_t buffer_index = 0;
  for (size_t i = 0; i < position.size(); i++) {
    buffer_index += strides_[i] * position[i];
  }
  return buffer_data()[buffer_index];
}

template <value_type T>
inline T &array<T>::at(const std::vector<size_t> &position) {
  // TODO: bounds_check_(position);
  size_t buffer_index = 0;
  for (size_t i = 0; i < position.size(); i++) {
    buffer_index += strides_[i] * position[i];
  }
  return buffer_data()[buffer_index];
}

template <value_type T>
template <size_t I>
inline auto array<T>::take() const {
  if constexpr (I == 0) {
    return std::tuple<>();
  } else {
    auto t = take<I - 1>();
    return std::tuple_cat(t, std::tuple<T>(at(I - 1)));
  }
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::operator[](size_t row) const {
  if (dimension() == 0 || row >= shape_[0]) {
    throw std::runtime_error("array: row is out of bounds.");
  }

  array tmp(*this);

  auto s = shape();
  s.erase(s.begin());
  tmp.reshape(s);

  auto stride = strides_[0];
  tmp.storage_.off = storage_.off + stride * row;
  tmp.storage_.len = stride;
  return tmp;
}

//----------------------------------------------------------------------------

template <value_type T>
class element_iterator {
 public:
  using difference_type = std::ptrdiff_t;
  using reference = T &;
  using iterator_concept = std::forward_iterator_tag;

  element_iterator(array<T> *arr, size_t i) : arr_(arr), i_(i) {}

  element_iterator &operator++() {
    ++i_;
    return *this;
  }

  element_iterator operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  reference &operator*() { return arr_->at(i_); }

  friend bool operator==(const element_iterator &a, const element_iterator &b) {
    return a.i_ == b.i_;
  };

 private:
  array<T> *arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T>
class const_element_iterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using iterator_concept = std::forward_iterator_tag;

  const_element_iterator(const array<T> *arr, size_t i) : arr_(arr), i_(i) {}

  const_element_iterator &operator++() {
    ++i_;
    return *this;
  }

  const_element_iterator operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  value_type operator*() const { return arr_->at(i_); }

  friend bool operator==(const const_element_iterator &a,
                         const const_element_iterator &b) {
    return a.i_ == b.i_;
  };

 private:
  const array<T> *arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T>
struct element_range {
  element_range(array<T> *arr) : arr_(arr) {}
  auto begin() { return element_iterator(arr_, 0); }
  auto end() { return element_iterator(arr_, arr_->element_count()); }
  array<T> *arr_ = nullptr;
};

template <value_type T>
struct const_element_range {
  const_element_range(const array<T> *arr) : arr_(arr) {}
  auto begin() { return const_element_iterator(arr_, 0); }
  auto end() { return const_element_iterator(arr_, arr_->element_count()); }
  auto cbegin() const { return const_element_iterator(arr_, 0); }
  auto cend() const {
    return const_element_iterator(arr_, arr_->element_count());
  }
  const array<T> *arr_ = nullptr;
};

template <value_type T>
inline auto array<T>::element_begin() {
  return element_iterator(this, 0);
}

template <value_type T>
inline auto array<T>::element_end() {
  return element_iterator(this, element_count());
}

template <value_type T>
inline auto array<T>::element_cbegin() const {
  return const_element_iterator(this, 0);
}

template <value_type T>
inline auto array<T>::element_cend() const {
  return const_element_iterator(this, element_count());
}

template <value_type T>
inline auto array<T>::elements() {
  return element_range(this);
}

template <value_type T>
inline auto array<T>::elements() const {
  return const_element_range(this);
}

//----------------------------------------------------------------------------

template <value_type T>
class row_iterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = array<T>;
  using iterator_concept = std::forward_iterator_tag;

  row_iterator(array<T> *arr, size_t i) : arr_(arr), i_(i) {}

  row_iterator &operator++() {
    ++i_;
    return *this;
  }

  row_iterator operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  value_type operator*() { return (*arr_)[i_]; }

  friend bool operator==(const row_iterator &a, const row_iterator &b) {
    return a.i_ == b.i_;
  };

 private:
  array<T> *arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T>
class const_row_iterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = array<T>;
  using iterator_concept = std::forward_iterator_tag;

  const_row_iterator(const array<T> *arr, size_t i) : arr_(arr), i_(i) {}

  const_row_iterator &operator++() {
    ++i_;
    return *this;
  }

  const_row_iterator operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  value_type operator*() const { return (*arr_)[i_]; }

  friend bool operator==(const const_row_iterator &a,
                         const const_row_iterator &b) {
    return a.i_ == b.i_;
  };

 private:
  const array<T> *arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T, size_t N>
class row_tuple_iterator {
 public:
  using difference_type = std::ptrdiff_t;
  using reference = array<T> &;
  using iterator_concept = std::forward_iterator_tag;

  row_tuple_iterator(array<T> *arr, size_t i) : arr_(arr), i_(i) {}

  row_tuple_iterator &operator++() {
    ++i_;
    return *this;
  }

  row_tuple_iterator operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  auto operator*() const { return (*arr_)[i_].template take<N>(); }

  friend bool operator==(const row_tuple_iterator &a,
                         const row_tuple_iterator &b) {
    return a.i_ == b.i_;
  };

 private:
  array<T> *arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T, size_t N>
class const_row_tuple_iterator {
 public:
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::forward_iterator_tag;

  const_row_tuple_iterator(const array<T> *arr, size_t i) : arr_(arr), i_(i) {}

  const_row_tuple_iterator &operator++() {
    ++i_;
    return *this;
  }

  const_row_tuple_iterator operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  auto operator*() const { return (*arr_)(i_).template take<N>(); }

  friend bool operator==(const const_row_tuple_iterator &a,
                         const const_row_tuple_iterator &b) {
    return a.i_ == b.i_;
  };

 private:
  const array<T> *arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T>
struct row_range {
  row_range(array<T> *arr) : arr_(arr) {}
  auto begin() { return row_iterator(arr_, 0); }
  auto end() { return row_iterator(arr_, arr_->shape()[0]); }
  array<T> *arr_ = nullptr;
};

template <value_type T>
struct const_row_range {
  const_row_range(const array<T> *arr) : arr_(arr) {}
  auto begin() const { return const_row_iterator(arr_, 0); }
  auto end() const { return const_row_iterator(arr_, arr_->shape()[0]); }
  auto cbegin() const { return const_row_iterator(arr_, 0); }
  auto cend() const { return const_row_iterator(arr_, arr_->shape()[0]); }
  const array<T> *arr_ = nullptr;
};

template <value_type T, size_t N>
struct row_tuple_range {
  row_tuple_range(array<T> *arr) : arr_(arr) {}
  auto begin() { return row_tuple_iterator<T, N>(arr_, 0); }
  auto end() { return row_tuple_iterator<T, N>(arr_, arr_->shape()[0]); }
  array<T> *arr_ = nullptr;
};

template <value_type T, size_t N>
struct const_row_tuple_range {
  const_row_tuple_range(array<T> *arr) : arr_(arr) {}
  auto cbegin() const { return const_row_tuple_iterator<T, N>(arr_, 0); }
  auto cend() const {
    return const_row_tuple_iterator<T, N>(arr_, arr_->shape()[0]);
  }
  const array<T> *arr_ = nullptr;
};

template <value_type T>
inline auto array<T>::begin() {
  return row_iterator(this, 0);
}

template <value_type T>
inline auto array<T>::end() {
  return row_iterator(this, shape_[0]);
}

template <value_type T>
inline auto array<T>::begin() const {
  return const_row_iterator(this, 0);
}

template <value_type T>
inline auto array<T>::end() const {
  return const_row_iterator(this, shape_[0]);
}

template <value_type T>
inline auto array<T>::cbegin() const {
  return const_row_iterator(this, 0);
}
template <value_type T>
inline auto array<T>::cend() const {
  return const_row_iterator(this, shape_[0]);
}

template <value_type T>
template <size_t N>
inline auto array<T>::rows() {
  if constexpr (N == 0) {
    return row_range(this);
  } else {
    return row_tuple_range<T, N>(this);
  }
}

template <value_type T>
template <size_t N>
inline auto array<T>::rows() const {
  if constexpr (N == 0) {
    return const_row_range(this);
  } else {
    return const_row_tuple_range<T, N>(this);
  }
}

//----------------------------------------------------------------------------

template <value_type T>
inline void array<T>::set(std::input_iterator auto it) {
  // TODO: parallel operation on GPU
  for (size_t i = 0; i < element_count(); i++) {
    at(i) = *it++;
  }
}

template <value_type T>
inline void array<T>::set(std::initializer_list<T> l) {
  std::ranges::copy(l, element_begin());
}

//----------------------------------------------------------------------------

template <value_type T>
inline void array<T>::constants(T val) {
  std::fill(buffer_data(), buffer_data() + buffer_element_count(), val);
}

template <value_type T>
inline void array<T>::zeros() {
  constants(0);
};

template <value_type T>
inline void array<T>::ones() {
  constants(1);
};

template <value_type T>
inline void array<T>::random() {
  std::generate(buffer_data(), buffer_data() + buffer_element_count(), []() {
    return static_cast<float>(static_cast<double>(rand()) / RAND_MAX);
  });
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::operator+(const array &rhs) const {
  return arithmetic_operation_(*this, rhs, ArithmeticOperation::Add);
}

template <value_type T>
inline array<T> array<T>::operator-(const array &rhs) const {
  return arithmetic_operation_(*this, rhs, ArithmeticOperation::Sub);
}

template <value_type T>
inline array<T> array<T>::operator*(const array &rhs) const {
  return arithmetic_operation_(*this, rhs, ArithmeticOperation::Mul);
}

template <value_type T>
inline array<T> array<T>::operator/(const array &rhs) const {
  return arithmetic_operation_(*this, rhs, ArithmeticOperation::Div);
}

template <value_type T>
inline array<T> array<T>::pow(const array &rhs) const {
  return arithmetic_operation_(*this, rhs, ArithmeticOperation::Pow);
}

template <value_type T>
inline void array<T>::operator+=(const array &rhs) {
  // TODO: in-place
  *this = arithmetic_operation_(*this, rhs, ArithmeticOperation::Add);
}

template <value_type T>
inline void array<T>::operator-=(const array &rhs) {
  // TODO: in-place
  *this = arithmetic_operation_(*this, rhs, ArithmeticOperation::Sub);
}

template <value_type T>
inline void array<T>::operator*=(const array &rhs) {
  // TODO: in-place
  *this = arithmetic_operation_(*this, rhs, ArithmeticOperation::Mul);
}

template <value_type T>
inline void array<T>::operator/=(const array &rhs) {
  // TODO: in-place
  *this = arithmetic_operation_(*this, rhs, ArithmeticOperation::Div);
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::dot(const array &rhs) const {
  switch (device_) {
    case Device::GPU:
      return dot_operation_(rhs, gpu_dot_operation_);
    case Device::CPU:
      return dot_operation_(rhs, cpu_dot_operation_);
  }
}

template <value_type T>
inline array<T> array<T>::linear(const array& W, const array& b) const {
  return dot(W) + b;
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<float> array<T>::sigmoid() const {
  // TODO: parallel operation on GPU
  auto tmp = array<float>(shape_, 0.0);
  for (size_t i = 0; i < element_count(); i++) {
    tmp.at(i) = 1.0 / (1.0 + std::exp(-static_cast<float>(at(i))));
  }
  return tmp;
}

//----------------------------------------------------------------------------

template <value_type T>
inline T array<T>::sum() const {
  return std::accumulate(element_cbegin(), element_cend(), T{});
}

template <value_type T>
inline array<T> array<T>::sum(size_t axis) const {
  auto s = shape_;
  s.erase(s.begin() + axis);

  auto tmp = array(s, T{});

  enumerate_position_([&](const auto &pos) {
    auto p = pos;
    p.erase(p.begin() + axis);

    tmp.at(p) += at(pos);
  });

  return tmp;
}

template <value_type T>
inline float array<T>::mean() const {
  return sum() / static_cast<float>(element_count());
}

template <value_type T>
inline array<float> array<T>::mean(size_t axis) const {
  auto t = sum(axis);
  if constexpr (std::is_same_v<T, float>) {
    return t / shape_[axis];
  } else {
    return t.template clone<float>() / shape_[axis];
  }
}

template <value_type T>
inline T array<T>::min() const {
  T min_val = std::numeric_limits<T>::max();
  for (size_t i = 0; i < buffer_element_count(); i++) {
    auto val = buffer_data()[i];
    if (val < min_val) {
      min_val = val;
    }
  }
  return min_val;
}

template <value_type T>
inline T array<T>::max() const {
  T max_val = std::numeric_limits<T>::min();
  for (size_t i = 0; i < buffer_element_count(); i++) {
    auto val = buffer_data()[i];
    if (val > max_val) {
      max_val = val;
    }
  }
  return max_val;
}

template <value_type T>
inline size_t array<T>::count() const {
  size_t cnt = 0;
  for (size_t i = 0; i < element_count(); i++) {
    if (at(i)) {
      cnt++;
    }
  }
  return cnt;
}

template <value_type T>
inline bool array<T>::all(arithmetic auto val) const {
  for (size_t i = 0; i < buffer_element_count(); i++) {
    if (buffer_data()[i] != val) {
      return false;
    }
  }
  return true;
}

template <value_type T>
template <typename U>
inline bool array<T>::all(U fn) const {
  for (size_t i = 0; i < buffer_element_count(); i++) {
    if (!fn(buffer_data()[i])) {
      return false;
    }
  }
  return true;
}

template <value_type T>
inline array<float> array<T>::softmax() const {
  if (dimension() == 1) {
    auto c = min();
    auto tmp = array<float>(shape_, 0.0);

    for (size_t i = 0; i < element_count(); i++) {
      tmp.at(i) = std::exp(at(i) - c);
    }
    return tmp / tmp.sum();
  } else if (dimension() == 2) {
    auto tmp = array<float>(shape_, 0.0);

    for (size_t i = 0; i < shape_[0]; i++) {
      const auto row = (*this)[i];
      auto c = row.min();
      for (size_t j = 0; j < row.element_count(); j++) {
        tmp[i].at(j) = std::exp(row.at(j) - c);
      }
      auto smax = tmp[i] / tmp[i].sum();

      for (size_t j = 0; j < row.element_count(); j++) {
        tmp[i].at(j) = smax.at(j);
      }
    }
    return tmp;
  }

  throw std::runtime_error(
      "array: softmax is available only for 1 or 2 dimension array.");
}

template <value_type T>
inline auto array<T>::argmax() const {
  if (dimension() == 2) {
    auto row_count = shape_[0];
    auto tmp = array<int>({row_count}, 0);

    for (size_t i = 0; i < row_count; i++) {
      const auto row = (*this)[i];

      size_t max_index = 0;
      {
        T max_val = std::numeric_limits<T>::min();
        for (size_t j = 0; j < row.buffer_element_count(); j++) {
          auto val = row.buffer_data()[j];
          if (val > max_val) {
            max_val = val;
            max_index = j;
          }
        }
      }

      tmp.at(i) = max_index;
    }
    return tmp;
  }

  throw std::runtime_error("array: argmax is available for 2 dimension array.");
}

template <value_type T>
inline float array<T>::mean_square_error(const array &rhs) const {
  return (*this - rhs).pow(2).mean();
}

//----------------------------------------------------------------------------

template <value_type T>
inline std::string array<T>::print_shape_type(const shape_type &shape) const {
  std::stringstream ss;
  ss << "{";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i != 0) {
      ss << ", ";
    }
    ss << shape[i];
  }
  ss << "}";
  return ss.str();
}

template <value_type T>
inline std::string array<T>::print_shape() const {
  return print_shape_type(shape_);
}

template <value_type T>
inline std::string array<T>::print_strides() const {
  return print_shape_type(strides_);
}

template <value_type T>
inline std::string array<T>::print_data_type() const {
  if constexpr (std::is_same_v<T, float>) {
    return "float";
  } else {
    return "int";
  }
}

template <value_type T>
inline std::string array<T>::print_info() const {
  std::stringstream ss;
  ss << "dtype: " << print_data_type() << ", dim: " << dimension()
     << ", shape: " << print_shape() << ", strides: " << print_strides();
  return ss.str();
}

template <value_type T>
inline std::string array<T>::print_array() const {
  auto loop = [&](auto self, auto &os, auto dim, auto arr_index) {
    auto n = shape_[dim];
    if (dim + 1 == dimension()) {
      for (size_t i = 0; i < n; i++, arr_index++) {
        if (i > 0) {
          os << ", ";
        }
        if constexpr (std::is_same_v<T, bool>) {
          os << (at(arr_index) ? "true" : "false");
        } else {
          os << at(arr_index);
        }
      }
      return arr_index;
    }

    for (size_t i = 0; i < n; i++) {
      if (dim < dimension() && i > 0) {
        os << ",\n";
        if (dimension() >= 3 && dim == 0 && i > 0) {
          os << "\n";
        }
        for (size_t j = 0; j <= dim; j++) {
          os << " ";
        }
      }
      os << '{';
      arr_index = self(self, os, dim + 1, arr_index);
      os << '}';
    }
    return arr_index;
  };

  std::stringstream ss;
  if (dimension() == 0) {
    ss << at();
  } else {
    ss << '{';
    loop(loop, ss, 0, 0);
    ss << '}';
  }
  return ss.str();
}

//----------------------------------------------------------------------------

template <value_type T>
inline void array<T>::allocate_buffer_() {
  storage_.off = 0;
  storage_.len = element_count();
  storage_.buf = metal::default_device().make_buffer(storage_.len * sizeof(T));
}

//----------------------------------------------------------------------------

template <typename T>
constexpr size_t nested_initializer_item_count_(const T &l) {
  return 1;
}

template <typename T>
constexpr size_t nested_initializer_item_count_(std::initializer_list<T> l) {
  size_t count = 0;
  for (auto it = l.begin(); it != l.end(); ++it) {
    count += nested_initializer_item_count_(*it);
  }
  return count;
}

template <typename T>
constexpr void nested_initializer_copy_(T &&dst, const auto &src) {
  *dst++ = src;
}

template <typename T, typename U>
constexpr void nested_initializer_copy_(T &&dst, std::initializer_list<U> src) {
  for (auto it = src.begin(); it != src.end(); ++it) {
    nested_initializer_copy_(std::forward<T>(dst), *it);
  }
}

template <value_type T>
inline void array<T>::copy_initializer_list_(const auto &l) {
  if (nested_initializer_item_count_(l) != element_count()) {
    throw std::runtime_error("array: invalid initializer list.");
  }
  nested_initializer_copy_(buffer_data(), l);
}

//----------------------------------------------------------------------------

template <value_type T>
inline void array<T>::bounds_check_(size_t i) const {
  if (strides_.empty() || i >= element_count()) {
    throw std::runtime_error("array: index is out of bounds.");
  }
}

template <value_type T>
inline void array<T>::bounds_check_(size_t x, size_t y) const {
  if (dimension() != 2 || x >= shape_[0] || y >= shape_[1]) {
    throw std::runtime_error("array: (x, y) is out of bounds.");
  }
}

template <value_type T>
inline void array<T>::bounds_check_(size_t x, size_t y, size_t z) const {
  if (dimension() != 3 || x >= shape_[0] || y >= shape_[1] || z >= shape_[2]) {
    throw std::runtime_error("array: (x, y, z) is out of bounds.");
  }
}

//----------------------------------------------------------------------------

template <value_type T>
template <typename U>
inline void array<T>::enumerate_position_(size_t shape_index,
                                          std::vector<size_t> &position,
                                          U fn) const {
  if (shape_index == shape_.size()) {
    fn(position);
    return;
  }

  for (size_t i = 0; i < shape_[shape_index]; i++) {
    position[shape_index] = i;
    enumerate_position_(shape_index + 1, position, fn);
  }
}

template <value_type T>
template <typename U>
inline void array<T>::enumerate_position_(U fn) const {
  std::vector<size_t> position(shape_.size());
  for (size_t i = 0; i < shape_[0]; i++) {
    position[0] = i;
    enumerate_position_(1, position, fn);
  }
}

//----------------------------------------------------------------------------

template <value_type T>
inline auto array<T>::broadcast_(const array &lhs, const array &rhs, auto cb) {
  if (lhs.shape() == rhs.shape()) {
    return cb(lhs, rhs);
  } else if (lhs.dimension() < rhs.dimension()) {
    return cb(lhs.broadcast(rhs.shape()), rhs);
  } else if (lhs.dimension() > rhs.dimension()) {
    return cb(lhs, rhs.broadcast(lhs.shape()));
  }
  throw std::runtime_error("array: invalid operation.");
}

template <value_type T>
template <value_type U>
inline array<U> array<T>::apply_binary_operation_(const array &rhs,
                                                  auto ope) const {
  return broadcast_(*this, rhs, [ope](const auto &lhs, const auto &rhs) {
    // TODO: parallel operation on GPU
    auto tmp = array<U>(lhs.shape(), U{});
    for (size_t i = 0; i < lhs.element_count(); i++) {
      tmp.at(i) = ope(lhs.at(i), rhs.at(i));
    }
    return tmp;
  });
}

//----------------------------------------------------------------------------

template <value_type T>
inline auto array<T>::gpu_arithmetic_operation_(const array &lhs,
                                                const array &rhs,
                                                ArithmeticOperation ope) {
  return broadcast_(lhs, rhs, [ope](const auto &lhs, const auto &rhs) {
    auto tmp = array(lhs.shape(), T{});
    switch (ope) {
      case ArithmeticOperation::Add:
        metal::default_device().add<T>(lhs.storage_, rhs.storage_,
                                       tmp.storage_);
        break;
      case ArithmeticOperation::Sub:
        metal::default_device().sub<T>(lhs.storage_, rhs.storage_,
                                       tmp.storage_);
        break;
      case ArithmeticOperation::Mul:
        metal::default_device().mul<T>(lhs.storage_, rhs.storage_,
                                       tmp.storage_);
        break;
      case ArithmeticOperation::Div:
        metal::default_device().div<T>(lhs.storage_, rhs.storage_,
                                       tmp.storage_);
        break;
      case ArithmeticOperation::Pow:
        metal::default_device().pow<T>(lhs.storage_, rhs.storage_,
                                       tmp.storage_);
        break;
      default:
        assert(false);
        break;
    }
    return tmp;
  });
}

template <value_type T>
inline auto array<T>::cpu_arithmetic_operation_(const array &lhs,
                                                const array &rhs,
                                                ArithmeticOperation ope) {
  switch (ope) {
    case ArithmeticOperation::Add:
      return lhs.apply_binary_operation_(
          rhs, [](auto lhs, auto rhs) { return lhs + rhs; });
      break;
    case ArithmeticOperation::Sub:
      return lhs.apply_binary_operation_(
          rhs, [](auto lhs, auto rhs) { return lhs - rhs; });
      break;
    case ArithmeticOperation::Mul:
      return lhs.apply_binary_operation_(
          rhs, [](auto lhs, auto rhs) { return lhs * rhs; });
      break;
    case ArithmeticOperation::Div:
      return lhs.apply_binary_operation_(
          rhs, [](auto lhs, auto rhs) { return lhs / rhs; });
      break;
    case ArithmeticOperation::Pow:
      return lhs.apply_binary_operation_(
          rhs, [](auto lhs, auto rhs) { return std::pow(lhs, rhs); });
      break;
    default:
      assert(false);
      break;
  }
}

template <value_type T>
inline auto array<T>::arithmetic_operation_(const array &lhs, const array &rhs,
                                            ArithmeticOperation ope) {
  switch (device_) {
    case Device::GPU:
      return gpu_arithmetic_operation_(lhs, rhs, ope);
    case Device::CPU:
      return cpu_arithmetic_operation_(lhs, rhs, ope);
  }
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::cpu_dot_operation_(const array &lhs,
                                             const array &rhs) {
  auto rows = lhs.shape_[0];
  auto cols = rhs.shape_[1];
  auto m = lhs.shape_[1];
  auto tmp = array({rows, cols}, T{});

  for (size_t row = 0; row < rows; row++) {
    for (size_t col = 0; col < cols; col++) {
      T val{};
      for (size_t i = 0; i < m; i++) {
        val += lhs.at(row, i) * rhs.at(i, col);
      }
      tmp.at(row, col) = val;
    }
  }
  return tmp;
}

template <value_type T>
inline array<T> array<T>::gpu_dot_operation_(const array &lhs,
                                             const array &rhs) {
  auto tmp = array({lhs.shape_[0], rhs.shape_[1]}, T{});

  metal::default_device().dot<T>(lhs.storage_, rhs.storage_, tmp.storage_,
                                 lhs.shape_[1], lhs.shape_[0], rhs.shape_[1]);

  return tmp;
}

template <value_type T>
template <typename U>
inline array<T> array<T>::dot_operation_(const array &rhs, U fn) const {
  if (dimension() == 2 && rhs.dimension() == 2 && shape_[1] == rhs.shape_[0]) {
    return fn(*this, rhs);
  }

  if (dimension() == 1 && rhs.dimension() == 2 && shape_[0] == rhs.shape_[0]) {
    auto lhs2 = *this;
    lhs2.reshape({1, shape_[0]});

    auto tmp = fn(lhs2, rhs);
    tmp.reshape({rhs.shape_[1]});
    return tmp;
  }

  if (dimension() == 2 && rhs.dimension() == 1 && shape_[1] == rhs.shape_[0]) {
    auto rhs2 = rhs;
    rhs2.reshape({rhs.shape_[0], 1});

    auto tmp = fn(*this, rhs2);
    tmp.reshape({shape_[0]});
    return tmp;
  }

  if (dimension() == 1 && rhs.dimension() == 1 && shape_[0] == rhs.shape_[0]) {
    auto lhs2 = *this;
    lhs2.reshape({1, shape_[0]});

    auto rhs2 = rhs;
    rhs2.reshape({rhs.shape_[0], 1});

    auto tmp = fn(lhs2, rhs2);
    tmp.reshape({});
    return tmp;
  }

  throw std::runtime_error("array: can't do `dot` operation.");
}

//----------------------------------------------------------------------------

template <value_type T>
inline std::ostream &operator<<(std::ostream &os, const array<T> &arr) {
  os << arr.print_array();
  return os;
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> operator+(auto lhs, const array<T> &rhs) {
  return array<T>(static_cast<T>(lhs)) + rhs;
}

template <value_type T>
inline array<T> operator-(auto lhs, const array<T> &rhs) {
  return array<T>(static_cast<T>(lhs)) - rhs;
}

template <value_type T>
inline array<T> operator*(auto lhs, const array<T> &rhs) {
  return array<T>(static_cast<T>(lhs)) * rhs;
}

template <value_type T>
inline array<T> operator/(auto lhs, const array<T> &rhs) {
  return array<T>(static_cast<T>(lhs)) / rhs;
}

//----------------------------------------------------------------------------

template <value_type T, value_type U>
inline array<T> where(const array<U> &cond, T x, T y) {
  // TODO: parallel operation on GPU
  auto tmp = array<T>(cond.shape(), T{});
  for (size_t i = 0; i < cond.element_count(); i++) {
    tmp.at(i) = cond.at(i) ? x : y;
  }
  return tmp;
}

//----------------------------------------------------------------------------

template <value_type T>
inline bool array_equal(const array<T> &a, const array<T> &b) {
  if (&a != &b) {
    if (a.shape() != b.shape()) {
      return false;
    }

    for (size_t i = 0; i < a.element_count(); i++) {
      if (a.at(i) != b.at(i)) {
        return false;
      }
    }
  }
  return true;
}

template <std::floating_point T, std::floating_point U>
inline bool is_close(T a, U b, float tolerance) {
  return std::abs(static_cast<float>(a) - static_cast<float>(b)) <= tolerance;
}

template <std::integral T, std::integral U>
inline bool is_close(T a, U b) {
  return a == b;
}

template <value_type T>
inline bool allclose(const array<T> &a, const array<T> &b, float tolerance) {
  if (&a != &b) {
    if (a.shape() != b.shape()) {
      return false;
    }

    for (size_t i = 0; i < a.element_count(); i++) {
      if constexpr (std::is_same_v<T, float>) {
        if (std::abs(a.at(i) - b.at(i)) > tolerance) {
          return false;
        }
      } else {
        if (a.at(i) != b.at(i)) {
          return false;
        }
      }
    }
  }
  return true;
}

//----------------------------------------------------------------------------

template <value_type T>
inline auto empty(const shape_type &shape) {
  return array<T>(shape, T{});
}

template <value_type T>
inline auto zeros(const shape_type &shape) {
  return array<T>(shape, 0);
}

template <value_type T>
inline auto ones(const shape_type &shape) {
  return array<T>(shape, 1);
}

inline auto random(const shape_type &shape) {
  auto tmp = array<float>(shape, 0.0);
  tmp.random();
  return tmp;
}

};  // namespace mtl
