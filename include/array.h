#pragma once

#include <metal.h>

#include <concepts>
#include <iostream>
#include <iterator>
#include <limits>
#include <ranges>

namespace mtl {

using shape_type = std::vector<size_t>;
using strides_type = std::vector<size_t>;

//------------------------------------------------------------------------------

template <typename T, size_t I>
struct nested_initializer_list_ {
  using nested_type = nested_initializer_list_<T, I - 1>::type;
  using type = std::initializer_list<nested_type>;
};

template <typename T>
struct nested_initializer_list_<T, 0> {
  using type = T;
};

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

template <typename T, size_t I>
using nested_initializer_list = nested_initializer_list_<T, I>::type;

//------------------------------------------------------------------------------

template <value_type T>
class array {
 private:
  shape_type shape_;
  strides_type strides_;

  mtl::managed_ptr<MTL::Buffer> buf_;
  size_t buf_off_ = 0;
  size_t buf_len_ = 0;

 public:
  array() = default;
  array(array &&rhs) = default;
  array(const array &rhs) = default;
  array &operator=(const array &rhs) = default;

  array(const shape_type &shape, T val) {
    reshape(shape);
    allocate_buffer_();
    constants(val);
  }

  array(const shape_type &shape, std::input_iterator auto it) {
    reshape(shape);
    allocate_buffer_();
    set(it);
  }

  array(const shape_type &shape, std::ranges::input_range auto &&r) {
    reshape(shape);
    allocate_buffer_();
    set(r);
  }

  array(T val) : array(nullptr, shape_type({})) { *buffer_data() = val; }

  array(nested_initializer_list<T, 1> l)
      : array(nullptr, nested_initializer_list_shape_(l)) {
    copy_initializer_list_(l);
  }
  array(nested_initializer_list<T, 2> l)
      : array(nullptr, nested_initializer_list_shape_(l)) {
    copy_initializer_list_(l);
  }
  array(nested_initializer_list<T, 3> l)
      : array(nullptr, nested_initializer_list_shape_(l)) {
    copy_initializer_list_(l);
  }
  array(nested_initializer_list<T, 4> l)
      : array(nullptr, nested_initializer_list_shape_(l)) {
    copy_initializer_list_(l);
  }

  static array from_shape(const shape_type &shape) {
    return array<T>(nullptr, shape);
  }

  //----------------------------------------------------------------------------

  template <value_type U = T>
  array<U> clone() const {
    auto tmp = array<U>::from_shape(shape_);
    for (size_t i = 0; i < element_count(); i++) {
      tmp.at(i) = static_cast<U>(at(i));
    }
    return tmp;
  }

  //----------------------------------------------------------------------------

  array<bool> operator==(const array &rhs) const {
    auto tmp = array<bool>::from_shape(shape_);
    for (size_t i = 0; i < element_count(); i++) {
      tmp.at(i) = at(i) == rhs.at(i);
    }
    return tmp;
  }

  //----------------------------------------------------------------------------

  size_t buffer_element_count() const { return buf_len_; }

  size_t buffer_bytes() const { return buf_len_ * sizeof(T); }

  T *buffer_data() { return static_cast<T *>(buf_->contents()) + buf_off_; }

  const T *buffer_data() const {
    return static_cast<const T *>(buf_->contents()) + buf_off_;
  }

  //----------------------------------------------------------------------------

  size_t element_count() const {
    // TODO: cache
    size_t count = 1;
    for (auto n : shape_) {
      count *= n;
    }
    return count;
  }

  size_t length() const {
    if (shape_.empty()) {
      throw std::runtime_error("array: cannot call with a scalar value.");
    }
    return shape_[0];
  }

  size_t dimension() const { return shape_.size(); }

  const shape_type &shape() const { return shape_; }

  const strides_type &strides() const { return strides_; }

  void reshape(const shape_type &shape) {
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

  const auto broadcast(const shape_type &target_shape) const {
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

  //----------------------------------------------------------------------------

  T at() const { return *buffer_data(); }

  T &at() { return *buffer_data(); }

  T at(size_t i) const {
    bounds_check_(i);
    return buffer_data()[i % buffer_element_count()];
  }

  T &at(size_t i) {
    bounds_check_(i);
    return buffer_data()[i % buffer_element_count()];
  }

  T at(size_t x, size_t y) const {
    bounds_check_(x, y);
    return buffer_data()[strides_[0] * x + y];
  }

  T &at(size_t x, size_t y) {
    bounds_check_(x, y);
    return buffer_data()[strides_[0] * x + y];
  }

  T at(size_t x, size_t y, size_t z) const {
    bounds_check_(x, y, z);
    return buffer_data()[(strides_[0] * x) + (strides_[1] * y) + z];
  }

  T &at(size_t x, size_t y, size_t z) {
    bounds_check_(x, y, z);
    return buffer_data()[(strides_[0] * x) + (strides_[1] * y) + z];
  }

  T at(const std::vector<size_t> &position) const {
    // TODO: bounds_check_(position);
    size_t buffer_index = 0;
    for (size_t i = 0; i < position.size(); i++) {
      buffer_index += strides_[i] * position[i];
    }
    return buffer_data()[buffer_index];
  }

  T &at(const std::vector<size_t> &position) {
    // TODO: bounds_check_(position);
    size_t buffer_index = 0;
    for (size_t i = 0; i < position.size(); i++) {
      buffer_index += strides_[i] * position[i];
    }
    return buffer_data()[buffer_index];
  }

  template <size_t I>
  auto take() const {
    if constexpr (I == 0) {
      return std::tuple<>();
    } else {
      auto t = take<I - 1>();
      return std::tuple_cat(t, std::tuple<T>(at(I - 1)));
    }
  }

  //----------------------------------------------------------------------------

  array operator[](size_t row) const {
    if (dimension() == 0 || row >= shape_[0]) {
      throw std::runtime_error("array: row is out of bounds.");
    }

    array tmp(*this);

    auto s = shape();
    s.erase(s.begin());
    tmp.reshape(s);

    auto stride = strides_[0];
    tmp.buf_off_ = buf_off_ + stride * row;
    tmp.buf_len_ = stride;
    return tmp;
  }

  //----------------------------------------------------------------------------

  class element_iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using reference = T &;
    using iterator_concept = std::forward_iterator_tag;

    element_iterator(array *arr, size_t i) : arr_(arr), i_(i) {}

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

    friend bool operator==(const element_iterator &a,
                           const element_iterator &b) {
      return a.i_ == b.i_;
    };

   private:
    array *arr_ = nullptr;
    size_t i_ = 0;
  };

  class const_element_iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using iterator_concept = std::forward_iterator_tag;

    const_element_iterator(const array *arr, size_t i) : arr_(arr), i_(i) {}

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
    const array *arr_ = nullptr;
    size_t i_ = 0;
  };

  auto element_begin() { return element_iterator(this, 0); }
  auto element_end() { return element_iterator(this, element_count()); }

  auto element_cbegin() const { return const_element_iterator(this, 0); }
  auto element_cend() const {
    return const_element_iterator(this, element_count());
  }

  struct element_range {
    element_range(array *arr) : arr_(arr) {}
    auto begin() { return element_iterator(arr_, 0); }
    auto end() { return element_iterator(arr_, arr_->element_count()); }
    array *arr_ = nullptr;
  };

  struct const_element_range {
    const_element_range(const array *arr) : arr_(arr) {}
    auto begin() { return const_element_iterator(arr_, 0); }
    auto end() { return const_element_iterator(arr_, arr_->element_count()); }
    auto cbegin() const { return const_element_iterator(arr_, 0); }
    auto cend() const {
      return const_element_iterator(arr_, arr_->element_count());
    }
    const array *arr_ = nullptr;
  };

  auto elements() { return element_range(this); }
  auto elements() const { return const_element_range(this); }

  //----------------------------------------------------------------------------

  class row_iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = array<T>;
    using iterator_concept = std::forward_iterator_tag;

    row_iterator(array *arr, size_t i) : arr_(arr), i_(i) {}

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
    array *arr_ = nullptr;
    size_t i_ = 0;
  };

  class const_row_iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = array<T>;
    using iterator_concept = std::forward_iterator_tag;

    const_row_iterator(const array *arr, size_t i) : arr_(arr), i_(i) {}

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
    const array *arr_ = nullptr;
    size_t i_ = 0;
  };

  template <size_t N>
  class row_tuple_iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using reference = array<T> &;
    using iterator_concept = std::forward_iterator_tag;

    row_tuple_iterator(array *arr, size_t i) : arr_(arr), i_(i) {}

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
    array *arr_ = nullptr;
    size_t i_ = 0;
  };

  template <size_t N>
  class const_row_tuple_iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using iterator_concept = std::forward_iterator_tag;

    const_row_tuple_iterator(const array *arr, size_t i) : arr_(arr), i_(i) {}

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
    const array *arr_ = nullptr;
    size_t i_ = 0;
  };

  auto begin() { return row_iterator(this, 0); }
  auto end() { return row_iterator(this, shape_[0]); }

  auto begin() const { return const_row_iterator(this, 0); }
  auto end() const { return const_row_iterator(this, shape_[0]); }

  auto cbegin() const { return const_row_iterator(this, 0); }
  auto cend() const { return const_row_iterator(this, shape_[0]); }

  struct row_range {
    row_range(array *arr) : arr_(arr) {}
    auto begin() { return row_iterator(arr_, 0); }
    auto end() { return row_iterator(arr_, arr_->shape()[0]); }
    array *arr_ = nullptr;
  };

  struct const_row_range {
    const_row_range(const array *arr) : arr_(arr) {}
    auto begin() const { return const_row_iterator(arr_, 0); }
    auto end() const { return const_row_iterator(arr_, arr_->shape()[0]); }
    auto cbegin() const { return const_row_iterator(arr_, 0); }
    auto cend() const { return const_row_iterator(arr_, arr_->shape()[0]); }
    const array *arr_ = nullptr;
  };

  template <size_t N>
  struct row_tuple_range {
    row_tuple_range(array *arr) : arr_(arr) {}
    auto begin() { return row_tuple_iterator<N>(arr_, 0); }
    auto end() { return row_tuple_iterator<N>(arr_, arr_->shape()[0]); }
    array *arr_ = nullptr;
  };

  template <size_t N>
  struct const_row_tuple_range {
    const_row_tuple_range(array *arr) : arr_(arr) {}
    auto cbegin() const { return const_row_tuple_iterator<N>(arr_, 0); }
    auto cend() const {
      return const_row_tuple_iterator<N>(arr_, arr_->shape()[0]);
    }
    const array *arr_ = nullptr;
  };

  template <size_t N = 0>
  auto rows() {
    if constexpr (N == 0) {
      return row_range(this);
    } else {
      return row_tuple_range<N>(this);
    }
  }

  template <size_t N = 0>
  auto rows() const {
    if constexpr (N == 0) {
      return const_row_range(this);
    } else {
      return const_row_tuple_range<N>(this);
    }
  }

  //----------------------------------------------------------------------------

  void set(std::input_iterator auto it) {
    for (size_t i = 0; i < element_count(); i++) {
      at(i) = *it++;
    }
  }
  void set(std::ranges::input_range auto &&r) {
    std::ranges::copy(r, element_begin());
  }
  void set(std::initializer_list<T> l) {
    std::ranges::copy(l, element_begin());
  }

  //----------------------------------------------------------------------------

  friend array operator+(const array &lhs, const array &rhs) {
    return binary_operation_(lhs, rhs, Operation::Add);
  }

  friend array operator-(const array &lhs, const array &rhs) {
    return binary_operation_(lhs, rhs, Operation::Sub);
  }

  friend array operator*(const array &lhs, const array &rhs) {
    return binary_operation_(lhs, rhs, Operation::Mul);
  }

  friend array operator/(const array &lhs, const array &rhs) {
    return binary_operation_(lhs, rhs, Operation::Div);
  }

  //------------------------------------------------------------------------------

  friend array operator+(const array &lhs, auto rhs) {
    return lhs + array(static_cast<T>(rhs));
  }

  friend array operator+(auto lhs, const array &rhs) {
    return array(static_cast<T>(lhs)) + rhs;
  }

  friend array operator-(const array &lhs, auto rhs) {
    return lhs - array(static_cast<T>(rhs));
  }

  friend array operator-(auto lhs, const array &rhs) {
    return array(static_cast<T>(lhs)) - rhs;
  }

  friend array operator*(const array &lhs, auto rhs) {
    return lhs * array(static_cast<T>(rhs));
  }

  friend array operator*(auto lhs, const array &rhs) {
    return array(static_cast<T>(lhs)) * rhs;
  }

  friend array operator/(const array &lhs, auto rhs) {
    return lhs / array(static_cast<T>(rhs));
  }

  friend array operator/(auto lhs, const array &rhs) {
    return array(static_cast<T>(lhs)) / rhs;
  }

  //----------------------------------------------------------------------------

  std::string print_shape() const {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape_.size(); i++) {
      if (i != 0) {
        ss << ":";
      }
      ss << shape_[i];
    }
    ss << ")";
    return ss.str();
  }

  std::string print_data_type() const {
    if constexpr (std::is_same_v<T, float>) {
      return "<float>";
    } else {
      return "<int>";
    }
  }

  std::string print_info() const {
    std::stringstream ss;
    ss << "dimension: " << dimension() << ", shape: " << print_shape()
       << ", dtype: " << print_data_type();
    return ss.str();
  }

  std::string print_array_numpy_format() const {
    std::string delims = " []";
    return print_array_(delims);
  }

  std::string print_array_cpp_format() const {
    std::string delims = ",{}";
    return print_array_(delims);
  }

  //----------------------------------------------------------------------------

  void constants(T val) {
    std::fill(buffer_data(), buffer_data() + buffer_element_count(), val);
  }

  void zeros() { constants(0); };

  void ones() { constants(1); };

  void random() {
    std::generate(buffer_data(), buffer_data() + buffer_element_count(), []() {
      return static_cast<float>(static_cast<double>(rand()) / RAND_MAX);
    });
  }

  //----------------------------------------------------------------------------

  array dot(const array &rhs) const {
    if (dimension() == 1 && rhs.dimension() == 1 &&
        shape_[0] == rhs.shape_[0]) {
      auto tmp = array::from_shape(shape_type{});

      T val = 0;
      for (size_t i = 0; i < shape_[0]; i++) {
        val += at(i) * rhs.at(i);
      }
      tmp.at() = val;
      return tmp;
    }

    if (dimension() == 2 && rhs.dimension() == 2 &&
        shape_[1] == rhs.shape_[0]) {
      auto rows = shape_[0];
      auto cols = rhs.shape_[1];
      auto tmp = array::from_shape(shape_type{rows, cols});

      for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
          T val = 0;
          for (size_t i = 0; i < shape_[1]; i++) {
            val += at(row, i) * rhs.at(i, col);
          }
          tmp.at(row, col) = val;
        }
      }
      return tmp;
    }

    if (dimension() == 1 && rhs.dimension() == 2 &&
        shape_[0] == rhs.shape_[0]) {
      auto rows = 1;
      auto cols = rhs.shape_[1];
      auto tmp = array::from_shape(shape_type{cols});

      for (size_t col = 0; col < cols; col++) {
        T val = 0;
        for (size_t i = 0; i < shape_[0]; i++) {
          val += at(i) * rhs.at(i, col);
        }
        tmp.at(col) = val;
      }
      return tmp;
    }

    if (dimension() == 2 && rhs.dimension() == 1 &&
        shape_[1] == rhs.shape_[0]) {
      auto rows = shape_[0];
      auto tmp = array::from_shape(shape_type{rows});

      for (size_t row = 0; row < rows; row++) {
        T val = 0;
        for (size_t i = 0; i < shape_[1]; i++) {
          val += at(row, i) * rhs.at(i);
        }
        tmp.at(row) = val;
      }
      return tmp;
    }

    throw std::runtime_error("array: can't do `dot` operation.");
  }

  //----------------------------------------------------------------------------

  array transpose() const {
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

  array sigmoid() const {
    auto tmp = array<float>::from_shape(shape_);
    for (size_t i = 0; i < element_count(); i++) {
      tmp.at(i) = 1.0 / (1.0 + std::exp(-static_cast<float>(at(i))));
    }
    return tmp;
  }

  //----------------------------------------------------------------------------

  float sum() const {
    return std::accumulate(element_cbegin(), element_cend(), T{});
  }

  array sum(size_t axis) const {
    auto s = shape_;
    s.erase(s.begin() + axis);

    auto tmp = array::from_shape(s);

    enumerate_position_([&](const auto &pos) {
      auto p = pos;
      p.erase(p.begin() + axis);

      tmp.at(p) += at(pos);
    });

    return tmp;
  }

  float mean() const { return sum() / static_cast<float>(element_count()); }

  array<float> mean(size_t axis) const {
    auto t = sum(axis);
    if constexpr (std::is_same_v<T, float>) {
      return t / shape_[axis];
    } else {
      return t.template clone<float>() / shape_[axis];
    }
  }

  T min() const {
    T min_val = std::numeric_limits<T>::max();
    for (size_t i = 0; i < buffer_element_count(); i++) {
      auto val = buffer_data()[i];
      if (val < min_val) {
        min_val = val;
      }
    }
    return min_val;
  }

  T max() const {
    T max_val = std::numeric_limits<T>::min();
    for (size_t i = 0; i < buffer_element_count(); i++) {
      auto val = buffer_data()[i];
      if (val > max_val) {
        max_val = val;
      }
    }
    return max_val;
  }

  template <typename U>
  bool all(U fn) const {
    for (size_t i = 0; i < buffer_element_count(); i++) {
      auto val = buffer_data()[i];
      if (!fn(val)) {
        return false;
      }
    }
    return true;
  }

  size_t count() const {
    size_t cnt = 0;
    for (size_t i = 0; i < element_count(); i++) {
      if (at(i)) {
        cnt++;
      }
    }
    return cnt;
  }

  array<float> softmax() const {
    if (dimension() == 1) {
      auto c = min();
      auto tmp = array<float>::from_shape(shape_);

      for (size_t i = 0; i < element_count(); i++) {
        tmp.at(i) = std::exp(at(i) - c);
      }
      return tmp / tmp.sum();
    } else if (dimension() == 2) {
      auto tmp = array<float>::from_shape(shape_);

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

  auto argmax() const {
    if (dimension() == 2) {
      auto row_count = shape_[0];
      auto tmp = array<int>::from_shape(shape_type{row_count});

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

    throw std::runtime_error(
        "array: argmax is available for 2 dimension array.");
  }

 private:
  array(void *dummy, const shape_type &shape) {
    reshape(shape);
    allocate_buffer_();
  }

  void allocate_buffer_() {
    buf_off_ = 0;
    buf_len_ = element_count();
    buf_ = mtl::newBuffer(buf_len_ * sizeof(T));
  }

  void copy_initializer_list_(const auto &l) {
    if (nested_initializer_item_count_(l) != element_count()) {
      throw std::runtime_error("array: invalid initializer list.");
    }
    nested_initializer_copy_(buffer_data(), l);
  }

  //----------------------------------------------------------------------------

  void bounds_check_(size_t i) const {
    if (strides_.empty() || i >= element_count()) {
      throw std::runtime_error("array: index is out of bounds.");
    }
  }

  void bounds_check_(size_t x, size_t y) const {
    if (dimension() != 2 || x >= shape_[0] || y >= shape_[1]) {
      throw std::runtime_error("array: (x, y) is out of bounds.");
    }
  }

  void bounds_check_(size_t x, size_t y, size_t z) const {
    if (dimension() != 3 || x >= shape_[0] || y >= shape_[1] ||
        z >= shape_[2]) {
      throw std::runtime_error("array: (x, y, z) is out of bounds.");
    }
  }

  //----------------------------------------------------------------------------

  template <typename U>
  void enumerate_position_(size_t shape_index, std::vector<size_t> &position,
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

  template <typename U>
  void enumerate_position_(U fn) const {
    std::vector<size_t> position(shape_.size());
    for (size_t i = 0; i < shape_[0]; i++) {
      position[0] = i;
      enumerate_position_(1, position, fn);
    }
  }

  //----------------------------------------------------------------------------

  auto print_array_(std::ostream &os, size_t dim, size_t arr_index,
                    const std::string &delims) const {
    auto n = shape_[dim];
    if (dim + 1 == dimension()) {
      for (size_t i = 0; i < n; i++, arr_index++) {
        if (i > 0) {
          os << delims[0];
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
        os << "\n ";
        for (size_t j = 0; j < dim; j++) {
          os << delims[0];
        }
      }
      os << delims[1];
      arr_index = print_array_(os, dim + 1, arr_index, delims);
      os << delims[2];
    }
    return arr_index;
  }

  auto print_array_(const std::string &delims) const {
    std::stringstream ss;
    if (dimension() == 0) {
      ss << at();
    } else {
      ss << delims[1];
      print_array_(ss, 0, 0, delims);
      ss << delims[2];
    }
    return ss.str();
  }

  //----------------------------------------------------------------------------

  static auto broadcast_(const array &lhs, const array &rhs, auto cb) {
    if (lhs.shape() == rhs.shape()) {
      return cb(lhs, rhs);
    } else if (lhs.dimension() < rhs.dimension()) {
      return cb(lhs.broadcast(rhs.shape()), rhs);
    } else if (lhs.dimension() > rhs.dimension()) {
      return cb(lhs, rhs.broadcast(lhs.shape()));
    }
    throw std::runtime_error("array: invalid operation.");
  }

  static auto gpu_binary_operation_(const array &lhs, const array &rhs,
                                    Operation ope) {
    return broadcast_(lhs, rhs, [ope](const auto &lhs, const auto &rhs) {
      auto tmp = array::from_shape(lhs.shape());
      mtl::compute<T>(
          lhs.buf_, lhs.buf_off_ * sizeof(T), lhs.buf_len_ * sizeof(T),
          rhs.buf_, rhs.buf_off_ * sizeof(T), rhs.buf_len_ * sizeof(T),
          tmp.buf_, tmp.buf_off_ * sizeof(T), tmp.buf_len_ * sizeof(T), ope);
      return tmp;
    });
  }

  static auto cpu_binary_operation_(const array &lhs, const array &rhs,
                                    Operation ope) {
    return broadcast_(lhs, rhs, [ope](const auto &lhs, const auto &rhs) {
      return [ope](auto cb) {
        switch (ope) {
          case Operation::Add:
            return cb([](T lhs, T rhs) { return lhs + rhs; });
          case Operation::Sub:
            return cb([](T lhs, T rhs) { return lhs - rhs; });
          case Operation::Mul:
            return cb([](T lhs, T rhs) { return lhs * rhs; });
          case Operation::Div:
            return cb([](T lhs, T rhs) { return lhs / rhs; });
        }
      }([&](auto fn) {
        auto tmp = array::from_shape(lhs.shape());
        for (size_t i = 0; i < lhs.element_count(); i++) {
          tmp.at(i) = fn(lhs.at(i), rhs.at(i));
        }
        return tmp;
      });
    });
  }

  //----------------------------------------------------------------------------

  static auto binary_operation_(const array &lhs, const array &rhs,
                                Operation ope) {
    switch (device) {
      case Device::GPU:
        return gpu_binary_operation_(lhs, rhs, ope);
      case Device::CPU:
        return cpu_binary_operation_(lhs, rhs, ope);
    }
  }
};

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

template <value_type T>
inline bool allclose(const array<T> &a, const array<T> &b,
                     float tolerance = 1e-3) {
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
inline std::ostream &operator<<(std::ostream &os, const array<T> &arr) {
  os << arr.print_array_numpy_format();
  return os;
}

//----------------------------------------------------------------------------

template <value_type T>
inline auto from_shape(const shape_type &shape) {
  return array<T>::from_shape(shape);
}

template <value_type T>
inline auto constants(const shape_type &shape, T val) {
  return array<T>(shape, val);
}

template <value_type T>
inline auto zeros(const shape_type &shape) {
  return array<T>(shape, 0);
}

template <value_type T>
inline auto ones(const shape_type &shape) {
  return array<T>(shape, 1);
}

template <value_type T>
inline auto random(const shape_type &shape) {
  auto tmp = array<T>::from_shape(shape);
  tmp.random();
  return tmp;
}

};  // namespace mtl
