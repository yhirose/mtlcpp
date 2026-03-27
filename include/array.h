#pragma once

#include <cpu.h>
#include <gpu.h>

#include <algorithm>
#include <concepts>
#include <format>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <sstream>

namespace sil {

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

using shape_type = std::vector<size_t>;
using strides_type = shape_type;

//------------------------------------------------------------------------------

template <value_type T, size_t I>
struct nested_initializer_list_ {
  using nested_type = nested_initializer_list_<T, I - 1>::type;
  using type = std::initializer_list<nested_type>;
};

template <value_type T>
struct nested_initializer_list_<T, 0> {
  using type = T;
};

template <value_type T, size_t I>
using nested_initializer_list = nested_initializer_list_<T, I>::type;

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

  auto *buffer_data(this auto &&self);
  auto buffer_span(this auto &&self);

  //----------------------------------------------------------------------------

  size_t element_count() const;
  size_t length() const;

  size_t dimension() const;
  const shape_type &shape() const;
  const strides_type &strides() const;

  void reshape(const shape_type &shape);

  const auto broadcast(const shape_type &target_shape) const;

  array transpose() const;

  //----------------------------------------------------------------------------

  auto &at(this auto &&self);
  auto &at(this auto &&self, size_t i);

  auto &operator[](this auto &&self, size_t x, size_t y);
  auto &operator[](this auto &&self, size_t x, size_t y, size_t z);

  auto &at(this auto &&self, const std::vector<size_t> &position);

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

  array linear(const array &W, const array &b) const;

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

  template <value_type U = T>
  array<U> one_hot(size_t class_count) const;

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
  storage storage_;

  //----------------------------------------------------------------------------

  void allocate_buffer_();

  static array make_uninit_(const shape_type &shape);

  array materialize_() const;

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

  static void cpu_arithmetic_dispatch_(const storage &lhs, const storage &rhs,
                                       storage &dst, ArithmeticOperation ope);

  static void msl_arithmetic_dispatch_(const storage &lhs, const storage &rhs,
                                       storage &dst, ArithmeticOperation ope);

  static auto cpu_arithmetic_operation_(const array &lhs, const array &rhs,
                                        ArithmeticOperation ope);

  static auto msl_arithmetic_operation_(const array &lhs, const array &rhs,
                                        ArithmeticOperation ope);

  static auto arithmetic_operation_(const array &lhs, const array &rhs,
                                    ArithmeticOperation ope);

  void arithmetic_inplace_(const array &rhs, ArithmeticOperation ope);

  void cpu_arithmetic_inplace_(const array &rhs, ArithmeticOperation ope);

  //----------------------------------------------------------------------------

  static array cpu_dot_operation_(const array &lhs, const array &rhs);
  static array mps_dot_operation_(const array &lhs, const array &rhs);
  static array auto_dot_operation_(const array &lhs, const array &rhs);
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
inline auto *array<T>::buffer_data(this auto &&self) {
  constexpr bool is_const =
      std::is_const_v<std::remove_reference_t<decltype(self)>>;
  using ptr_type = std::conditional_t<is_const, const T *, T *>;
  return static_cast<ptr_type>(self.storage_.data) + self.storage_.off;
}

template <value_type T>
inline auto array<T>::buffer_span(this auto &&self) {
  return std::span(self.buffer_data(), self.buffer_element_count());
}

//----------------------------------------------------------------------------

template <value_type T>
inline size_t array<T>::element_count() const {
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
  shape_ = shape;

  strides_.clear();
  strides_.push_back(1);
  for (int i = shape.size() - 1; i > 0; i--) {
    auto n = strides_.front() * shape[i];
    strides_.insert(strides_.begin(), n);
  }
}

template <value_type T>
inline const auto array<T>::broadcast(const shape_type &target_shape) const {
  if (target_shape.size() < dimension()) {
    throw std::runtime_error("array: invalid shape for broadcast.");
  }

  auto diff = target_shape.size() - dimension();
  for (size_t i = 0; i < dimension(); i++) {
    if (shape_[i] != target_shape[i + diff] && shape_[i] != 1) {
      throw std::runtime_error("array: invalid shape for broadcast.");
    }
  }

  if (target_shape == shape_) {
    return *this;
  }

  array tmp = *this;
  tmp.shape_ = target_shape;

  // Build strides: start from source strides, prepend zeros for new dims,
  // and zero out dims where source size is 1 (broadcast).
  tmp.strides_.resize(target_shape.size(), 0);
  for (size_t i = 0; i < target_shape.size(); i++) {
    int src_axis = static_cast<int>(i) - static_cast<int>(diff);
    if (src_axis < 0) {
      // New dimension added by higher-dim broadcast
      tmp.strides_[i] = 0;
    } else if (src_axis < static_cast<int>(dimension()) &&
               shape_[src_axis] == 1 && target_shape[i] != 1) {
      // Source dim is 1, broadcast to target size
      tmp.strides_[i] = 0;
    } else if (src_axis < static_cast<int>(dimension())) {
      // Keep original stride
      tmp.strides_[i] = strides_[src_axis];
    }
  }
  return tmp;
}

template <value_type T>
inline array<T> array<T>::transpose() const {
  if (dimension() == 1) {
    auto tmp = make_uninit_({1, element_count()});

    auto it = element_cbegin();
    for (size_t col = 0; col < element_count(); col++) {
      tmp[0, col] = *it;
      ++it;
    }
    return tmp;
  }

  if (dimension() == 2) {
    if (shape_[0] == 1) {
      auto tmp = make_uninit_({element_count()});

      auto it = element_cbegin();
      for (size_t row = 0; row < element_count(); row++) {
        tmp.at(row) = *it;
        ++it;
      }
      return tmp;
    } else {
      auto shape = shape_;
      std::ranges::reverse(shape);

      auto tmp = make_uninit_(shape);

      auto it = element_cbegin();
      for (size_t col = 0; col < shape[1]; col++) {
        for (size_t row = 0; row < shape[0]; row++) {
          tmp[row, col] = *it;
          ++it;
        }
      }
      return tmp;
    }
  }

  if (dimension() == 3) {
    auto shape = shape_;
    std::ranges::reverse(shape);

    auto tmp = make_uninit_(shape);

    auto it = element_cbegin();
    for (size_t z = 0; z < shape[2]; z++) {
      for (size_t y = 0; y < shape[1]; y++) {
        for (size_t x = 0; x < shape[0]; x++) {
          tmp[x, y, z] = *it;
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
inline auto &array<T>::at(this auto &&self) {
  return *self.buffer_data();
}

template <value_type T>
inline auto &array<T>::at(this auto &&self, size_t i) {
  self.bounds_check_(i);
  return self.buffer_data()[i % self.buffer_element_count()];
}

template <value_type T>
inline auto &array<T>::operator[](this auto &&self, size_t x, size_t y) {
  self.bounds_check_(x, y);
  return self.buffer_data()[self.strides_[0] * x + y];
}

template <value_type T>
inline auto &
array<T>::operator[](this auto &&self, size_t x, size_t y, size_t z) {
  self.bounds_check_(x, y, z);
  return self.buffer_data()[(self.strides_[0] * x) + (self.strides_[1] * y) +
                            z];
}

template <value_type T>
inline auto &array<T>::at(this auto &&self,
                          const std::vector<size_t> &position) {
  size_t buffer_index = 0;
  for (size_t i = 0; i < position.size(); i++) {
    buffer_index += self.strides_[i] * position[i];
  }
  return self.buffer_data()[buffer_index];
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

template <value_type T, bool IsConst>
class element_iterator_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;

 public:
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::forward_iterator_tag;

  element_iterator_(arr_ptr arr, size_t i) : arr_(arr), i_(i) {}

  element_iterator_ &operator++() {
    ++i_;
    return *this;
  }

  element_iterator_ operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  decltype(auto) operator*() const { return arr_->at(i_); }

  bool operator==(const element_iterator_ &) const = default;

 private:
  arr_ptr arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T>
using element_iterator = element_iterator_<T, false>;
template <value_type T>
using const_element_iterator = element_iterator_<T, true>;

template <value_type T, bool IsConst>
struct element_range_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;
  element_range_(arr_ptr arr) : arr_(arr) {}
  auto begin() const { return element_iterator_<T, IsConst>(arr_, 0); }
  auto end() const {
    return element_iterator_<T, IsConst>(arr_, arr_->element_count());
  }
  arr_ptr arr_ = nullptr;
};

template <value_type T>
using element_range = element_range_<T, false>;
template <value_type T>
using const_element_range = element_range_<T, true>;

template <value_type T>
inline auto array<T>::element_begin() {
  return element_iterator_<T, false>(this, 0);
}

template <value_type T>
inline auto array<T>::element_end() {
  return element_iterator_<T, false>(this, element_count());
}

template <value_type T>
inline auto array<T>::element_cbegin() const {
  return element_iterator_<T, true>(this, 0);
}

template <value_type T>
inline auto array<T>::element_cend() const {
  return element_iterator_<T, true>(this, element_count());
}

template <value_type T>
inline auto array<T>::elements() {
  return element_range_<T, false>(this);
}

template <value_type T>
inline auto array<T>::elements() const {
  return element_range_<T, true>(this);
}

//----------------------------------------------------------------------------

template <value_type T, bool IsConst>
class row_iterator_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;

 public:
  using difference_type = std::ptrdiff_t;
  using value_type = array<T>;
  using iterator_concept = std::forward_iterator_tag;

  row_iterator_(arr_ptr arr, size_t i) : arr_(arr), i_(i) {}

  row_iterator_ &operator++() {
    ++i_;
    return *this;
  }

  row_iterator_ operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  value_type operator*() const { return (*arr_)[i_]; }

  bool operator==(const row_iterator_ &) const = default;

 private:
  arr_ptr arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T>
using row_iterator = row_iterator_<T, false>;
template <value_type T>
using const_row_iterator = row_iterator_<T, true>;

template <value_type T, size_t N, bool IsConst>
class row_tuple_iterator_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;

 public:
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::forward_iterator_tag;

  row_tuple_iterator_(arr_ptr arr, size_t i) : arr_(arr), i_(i) {}

  row_tuple_iterator_ &operator++() {
    ++i_;
    return *this;
  }

  row_tuple_iterator_ operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  auto operator*() const { return (*arr_)[i_].template take<N>(); }

  bool operator==(const row_tuple_iterator_ &) const = default;

 private:
  arr_ptr arr_ = nullptr;
  size_t i_ = 0;
};

template <value_type T, size_t N>
using row_tuple_iterator = row_tuple_iterator_<T, N, false>;
template <value_type T, size_t N>
using const_row_tuple_iterator = row_tuple_iterator_<T, N, true>;

template <value_type T, bool IsConst>
struct row_range_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;
  row_range_(arr_ptr arr) : arr_(arr) {}
  auto begin() const { return row_iterator_<T, IsConst>(arr_, 0); }
  auto end() const {
    return row_iterator_<T, IsConst>(arr_, arr_->shape()[0]);
  }
  arr_ptr arr_ = nullptr;
};

template <value_type T>
using row_range = row_range_<T, false>;
template <value_type T>
using const_row_range = row_range_<T, true>;

template <value_type T, size_t N, bool IsConst>
struct row_tuple_range_ {
  using arr_ptr = std::conditional_t<IsConst, const array<T> *, array<T> *>;
  row_tuple_range_(arr_ptr arr) : arr_(arr) {}
  auto begin() const { return row_tuple_iterator_<T, N, IsConst>(arr_, 0); }
  auto end() const {
    return row_tuple_iterator_<T, N, IsConst>(arr_, arr_->shape()[0]);
  }
  arr_ptr arr_ = nullptr;
};

template <value_type T, size_t N>
using row_tuple_range = row_tuple_range_<T, N, false>;
template <value_type T, size_t N>
using const_row_tuple_range = row_tuple_range_<T, N, true>;

template <value_type T>
inline auto array<T>::begin() {
  return row_iterator_<T, false>(this, 0);
}

template <value_type T>
inline auto array<T>::end() {
  return row_iterator_<T, false>(this, shape_[0]);
}

template <value_type T>
inline auto array<T>::begin() const {
  return row_iterator_<T, true>(this, 0);
}

template <value_type T>
inline auto array<T>::end() const {
  return row_iterator_<T, true>(this, shape_[0]);
}

template <value_type T>
inline auto array<T>::cbegin() const {
  return begin();
}
template <value_type T>
inline auto array<T>::cend() const {
  return end();
}

template <value_type T>
template <size_t N>
inline auto array<T>::rows() {
  if constexpr (N == 0) {
    return row_range_<T, false>(this);
  } else {
    return row_tuple_range_<T, N, false>(this);
  }
}

template <value_type T>
template <size_t N>
inline auto array<T>::rows() const {
  if constexpr (N == 0) {
    return row_range_<T, true>(this);
  } else {
    return row_tuple_range_<T, N, true>(this);
  }
}

//----------------------------------------------------------------------------

template <value_type T>
inline void array<T>::set(std::input_iterator auto it) {

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
  std::ranges::fill(buffer_span(), val);
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
  thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::ranges::generate(buffer_span(), [&]() { return dist(gen); });
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
  arithmetic_inplace_(rhs, ArithmeticOperation::Add);
}

template <value_type T>
inline void array<T>::operator-=(const array &rhs) {
  arithmetic_inplace_(rhs, ArithmeticOperation::Sub);
}

template <value_type T>
inline void array<T>::operator*=(const array &rhs) {
  arithmetic_inplace_(rhs, ArithmeticOperation::Mul);
}

template <value_type T>
inline void array<T>::operator/=(const array &rhs) {
  arithmetic_inplace_(rhs, ArithmeticOperation::Div);
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::dot(const array &rhs) const {
  switch (device_) {
    case Device::MPS:
      return dot_operation_(rhs, mps_dot_operation_);
    case Device::Auto:
      return dot_operation_(rhs, auto_dot_operation_);
    case Device::CPU:
      return dot_operation_(rhs, cpu_dot_operation_);
  }
}

template <value_type T>
inline array<T> array<T>::linear(const array &W, const array &b) const {
  return dot(W) + b;
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<float> array<T>::sigmoid() const {
  if constexpr (std::is_same_v<T, float>) {
    if (device_ == Device::MPS ||
        (device_ == Device::Auto && element_count() >= 100'000)) {
      auto tmp = make_uninit_(shape_);
      gpu::sigmoid(storage_, tmp.storage_);
      return tmp;
    }
  }
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
  return *std::ranges::min_element(buffer_span());
}

template <value_type T>
inline T array<T>::max() const {
  return *std::ranges::max_element(buffer_span());
}

template <value_type T>
inline size_t array<T>::count() const {
  return std::ranges::count_if(buffer_span(), [](T v) { return !!v; });
}

template <value_type T>
inline bool array<T>::all(arithmetic auto val) const {
  return std::ranges::all_of(buffer_span(), [val](T v) { return v == val; });
}

template <value_type T>
template <typename U>
inline bool array<T>::all(U fn) const {
  return std::ranges::all_of(buffer_span(), fn);
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

template <value_type T>
template <value_type U>
inline array<U> array<T>::one_hot(size_t class_count) const {
  if (dimension() == 1) {
    auto tmp = array<U>({shape_[0], class_count}, U{});
    for (size_t i = 0; i < element_count(); i++) {
      tmp[i, at(i)] = 1;
    }
    return tmp;
  }

  throw std::runtime_error(
      "array: one_hot is available only for 1 dimension array.");
}

//----------------------------------------------------------------------------

template <value_type T>
inline std::string array<T>::print_shape_type(const shape_type &shape) const {
  std::string result = "{";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i != 0) result += ", ";
    result += std::format("{}", shape[i]);
  }
  result += "}";
  return result;
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
  return std::format("dtype: {}, dim: {}, shape: {}, strides: {}",
                     print_data_type(), dimension(), print_shape(),
                     print_strides());
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
inline array<T> array<T>::make_uninit_(const shape_type &shape) {
  array a;
  a.reshape(shape);
  a.allocate_buffer_();
  return a;
}

template <value_type T>
inline void array<T>::allocate_buffer_() {
  auto len = element_count();
  auto bytes = len * sizeof(T);

  storage_ = storage::make(bytes);
  storage_.off = 0;
  storage_.len = len;
}

template <value_type T>
inline array<T> array<T>::materialize_() const {
  if (buffer_element_count() == element_count()) {
    return *this;
  }

  auto tmp = make_uninit_(shape_);
  enumerate_position_([&](const auto &pos) { tmp.at(pos) = at(pos); });
  return tmp;
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
  } else if (lhs.dimension() == rhs.dimension()) {
    // Same dimension, check broadcast compatibility (each dim must be equal or 1)
    bool rhs_to_lhs = true, lhs_to_rhs = true;
    for (size_t i = 0; i < lhs.dimension(); i++) {
      if (lhs.shape()[i] != rhs.shape()[i]) {
        if (rhs.shape()[i] != 1) rhs_to_lhs = false;
        if (lhs.shape()[i] != 1) lhs_to_rhs = false;
      }
    }
    if (rhs_to_lhs) return cb(lhs, rhs.broadcast(lhs.shape()));
    if (lhs_to_rhs) return cb(lhs.broadcast(rhs.shape()), rhs);

    // Both sides need broadcast (e.g. {3,1} + {1,3} -> {3,3})
    shape_type target(lhs.dimension());
    bool compatible = true;
    for (size_t i = 0; i < lhs.dimension(); i++) {
      if (lhs.shape()[i] == rhs.shape()[i]) {
        target[i] = lhs.shape()[i];
      } else if (lhs.shape()[i] == 1) {
        target[i] = rhs.shape()[i];
      } else if (rhs.shape()[i] == 1) {
        target[i] = lhs.shape()[i];
      } else {
        compatible = false;
        break;
      }
    }
    if (compatible) {
      return cb(lhs.broadcast(target).materialize_(),
                rhs.broadcast(target).materialize_());
    }
  }
  throw std::runtime_error("array: invalid operation.");
}

template <value_type T>
template <value_type U>
inline array<U> array<T>::apply_binary_operation_(const array &rhs,
                                                  auto ope) const {
  return broadcast_(*this, rhs, [ope](const auto &lhs, const auto &rhs) {
  
    auto n = std::max(lhs.element_count(), rhs.element_count());
    auto shape = lhs.element_count() >= rhs.element_count() ? lhs.shape() : rhs.shape();
    auto tmp = array<U>(shape, U{});
    for (size_t i = 0; i < n; i++) {
      tmp.at(i) = ope(lhs.at(i % lhs.element_count()),
                       rhs.at(i % rhs.element_count()));
    }
    return tmp;
  });
}

//----------------------------------------------------------------------------

template <value_type T>
inline void array<T>::cpu_arithmetic_dispatch_(const storage &lhs,
                                               const storage &rhs,
                                               storage &dst,
                                               ArithmeticOperation ope) {
  switch (ope) {
    case ArithmeticOperation::Add: cpu::add<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Sub: cpu::sub<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Mul: cpu::mul<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Div: cpu::div<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Pow: cpu::pow<T>(lhs, rhs, dst); break;
  }
}

template <value_type T>
inline auto array<T>::cpu_arithmetic_operation_(const array &lhs,
                                                const array &rhs,
                                                ArithmeticOperation ope) {
  return broadcast_(lhs, rhs, [ope](const auto &lhs, const auto &rhs) {
    auto tmp = make_uninit_(lhs.shape());
    cpu_arithmetic_dispatch_(lhs.storage_, rhs.storage_, tmp.storage_, ope);
    return tmp;
  });
}

template <value_type T>
inline void array<T>::cpu_arithmetic_inplace_(const array &rhs,
                                                    ArithmeticOperation ope) {
  if (shape() == rhs.shape() || rhs.element_count() <= element_count()) {
    cpu_arithmetic_dispatch_(storage_, rhs.storage_, storage_, ope);
  } else {
    *this = cpu_arithmetic_operation_(*this, rhs, ope);
  }
}

template <value_type T>
inline void array<T>::msl_arithmetic_dispatch_(const storage &lhs,
                                               const storage &rhs,
                                               storage &dst,
                                               ArithmeticOperation ope) {
  switch (ope) {
    case ArithmeticOperation::Add: msl::add<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Sub: msl::sub<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Mul: msl::mul<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Div: msl::div<T>(lhs, rhs, dst); break;
    case ArithmeticOperation::Pow: msl::pow<T>(lhs, rhs, dst); break;
  }
}

template <value_type T>
inline auto array<T>::msl_arithmetic_operation_(const array &lhs,
                                                const array &rhs,
                                                ArithmeticOperation ope) {
  return broadcast_(lhs, rhs, [ope](const auto &lhs, const auto &rhs) {
    auto tmp = make_uninit_(lhs.shape());
    msl_arithmetic_dispatch_(lhs.storage_, rhs.storage_, tmp.storage_, ope);
    return tmp;
  });
}

template <value_type T>
inline auto array<T>::arithmetic_operation_(const array &lhs,
                                            const array &rhs,
                                            ArithmeticOperation ope) {
  switch (device_) {
    case Device::CPU:
      return cpu_arithmetic_operation_(lhs, rhs, ope);
    case Device::MPS:
      return msl_arithmetic_operation_(lhs, rhs, ope);
    case Device::Auto: {
      if constexpr (!std::same_as<T, float>) {
        return cpu_arithmetic_operation_(lhs, rhs, ope);
      } else {
        auto &bc = device_cache::instance();
        auto n = std::max(lhs.element_count(), rhs.element_count());
        auto b = device_cache::bucket(n);
        auto k = device_cache::key(static_cast<uint32_t>(ope), b);

        auto dispatch = [&](Device d) {
          return d == Device::CPU
                     ? cpu_arithmetic_operation_(lhs, rhs, ope)
                     : msl_arithmetic_operation_(lhs, rhs, ope);
        };

        Device device;
        if (bc.lookup(k, device)) {
          return dispatch(device);
        }

        // First time: measure both in alternating order to avoid bias
        array cpu_result, msl_result;

        auto cpu_time1 = device_cache::time([&] {
          cpu_result = cpu_arithmetic_operation_(lhs, rhs, ope);
        });
        auto msl_time1 = device_cache::time([&] {
          msl_result = msl_arithmetic_operation_(lhs, rhs, ope);
        });

        auto msl_time2 = device_cache::time([&] {
          msl_result = msl_arithmetic_operation_(lhs, rhs, ope);
        });
        auto cpu_time2 = device_cache::time([&] {
          cpu_result = cpu_arithmetic_operation_(lhs, rhs, ope);
        });

        auto cpu_time = std::min(cpu_time1, cpu_time2);
        auto msl_time = std::min(msl_time1, msl_time2);

        device = cpu_time <= msl_time ? Device::CPU : Device::MPS;
        bc.store(k, device);
        return device == Device::CPU ? cpu_result : msl_result;
      }
    }
  }
}

template <value_type T>
inline void array<T>::arithmetic_inplace_(const array &rhs,
                                          ArithmeticOperation ope) {
  switch (device_) {
    case Device::CPU:
      cpu_arithmetic_inplace_(rhs, ope);
      break;
    case Device::MPS:
      *this = msl_arithmetic_operation_(*this, rhs, ope);
      break;
    case Device::Auto: {
      if constexpr (!std::same_as<T, float>) {
        cpu_arithmetic_inplace_(rhs, ope);
      } else {
        auto &bc = device_cache::instance();
        auto n = std::max(element_count(), rhs.element_count());
        auto b = device_cache::bucket(n);
        auto k = device_cache::key(static_cast<uint32_t>(ope), b);

        Device device;
        if (bc.lookup(k, device)) {
          if (device == Device::MPS) {
            *this = msl_arithmetic_operation_(*this, rhs, ope);
          } else {
            cpu_arithmetic_inplace_(rhs, ope);
          }
        } else {
          // No cache entry yet, default to CPU
          cpu_arithmetic_inplace_(rhs, ope);
        }
      }
      break;
    }
  }
}

//----------------------------------------------------------------------------

template <value_type T>
inline array<T> array<T>::cpu_dot_operation_(const array &lhs,
                                             const array &rhs) {
  auto tmp = make_uninit_({lhs.shape_[0], rhs.shape_[1]});

  cpu::dot<T>(lhs.storage_, rhs.storage_, tmp.storage_, lhs.shape_[1],
              lhs.shape_[0], rhs.shape_[1]);

  return tmp;
}

template <value_type T>
inline array<T> array<T>::mps_dot_operation_(const array &lhs,
                                             const array &rhs) {
  if constexpr (std::same_as<T, float>) {
    auto tmp = make_uninit_({lhs.shape_[0], rhs.shape_[1]});
    mps::dot_f32(lhs.storage_, rhs.storage_, tmp.storage_,
                 lhs.shape_[1], lhs.shape_[0], rhs.shape_[1]);
    return tmp;
  } else {
    return cpu_dot_operation_(lhs, rhs);
  }
}

template <value_type T>
inline array<T> array<T>::auto_dot_operation_(const array &lhs,
                                              const array &rhs) {
  if constexpr (!std::same_as<T, float>) {
    return cpu_dot_operation_(lhs, rhs);
  } else {
    constexpr uint32_t dot_op_id = 100;  // distinct from ArithmeticOperation

    auto &bc = device_cache::instance();
    auto work = lhs.shape_[0] * lhs.shape_[1] * rhs.shape_[1];
    auto b = device_cache::bucket(work);
    auto k = device_cache::key(dot_op_id, b);

    auto dispatch = [&](Device d) {
      return d == Device::CPU
                 ? cpu_dot_operation_(lhs, rhs)
                 : mps_dot_operation_(lhs, rhs);
    };

    Device device;
    if (bc.lookup(k, device)) {
      return dispatch(device);
    }

    // First time: measure both in alternating order to avoid bias
    array cpu_result, mps_result;

    // Round 1: CPU then GPU
    auto cpu_time1 = device_cache::time([&] {
      cpu_result = cpu_dot_operation_(lhs, rhs);
    });
    auto mps_time1 = device_cache::time([&] {
      mps_result = mps_dot_operation_(lhs, rhs);
    });

    // Round 2: GPU then CPU
    auto mps_time2 = device_cache::time([&] {
      mps_result = mps_dot_operation_(lhs, rhs);
    });
    auto cpu_time2 = device_cache::time([&] {
      cpu_result = cpu_dot_operation_(lhs, rhs);
    });

    auto cpu_time = std::min(cpu_time1, cpu_time2);
    auto mps_time = std::min(mps_time1, mps_time2);

    device = cpu_time <= mps_time ? Device::CPU : Device::MPS;
    bc.store(k, device);
    return device == Device::CPU ? cpu_result : mps_result;
  }
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

  auto tmp = array<T>(cond.shape(), T{});
  for (size_t i = 0; i < cond.element_count(); i++) {
    tmp.at(i) = cond.at(i) ? x : y;
  }
  return tmp;
}

//----------------------------------------------------------------------------

template <value_type T>
inline bool array_equal(const array<T> &a, const array<T> &b) {
  if (&a == &b) return true;
  if (a.shape() != b.shape()) return false;
  for (size_t i = 0; i < a.element_count(); i++) {
    if (a.at(i) != b.at(i)) return false;
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
  if (&a == &b) return true;
  if (a.shape() != b.shape()) return false;
  auto as = a.buffer_span();
  auto bs = b.buffer_span();
  for (size_t i = 0; i < as.size(); i++) {
    if constexpr (std::is_same_v<T, float>) {
      if (std::abs(as[i] - bs[i]) > tolerance) return false;
    } else {
      if (as[i] != bs[i]) return false;
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

};  // namespace sil
