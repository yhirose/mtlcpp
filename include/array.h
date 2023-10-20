#pragma once

#include <metal.h>

#include <concepts>
#include <iterator>
#include <ranges>

namespace mtl {

using shape_type = std::vector<size_t>;
using strides_type = std::vector<size_t>;

//------------------------------------------------------------------------------

template <typename T, size_t I>
struct nested_initializer_list_ {
  using nested_type = typename nested_initializer_list_<T, I - 1>::type;
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
using nested_initializer_list = typename nested_initializer_list_<T, I>::type;

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
  array(array &&rhs) = default;
  array(const array &rhs) = default;             // TODO: use GPU
  array &operator=(const array &rhs) = default;  // TODO: use GPU

  array(const shape_type &shape) {
    reshape(shape);
    allocate_buffer_();
  }

  array(T val) : array(shape_type({})) { *buffer_data() = val; }

  array(nested_initializer_list<T, 1> l)
      : array(nested_initializer_list_shape_(l)) {
    copy_initializer_list_(l);
  }
  array(nested_initializer_list<T, 2> l)
      : array(nested_initializer_list_shape_(l)) {
    copy_initializer_list_(l);
  }
  array(nested_initializer_list<T, 3> l)
      : array(nested_initializer_list_shape_(l)) {
    copy_initializer_list_(l);
  }
  array(nested_initializer_list<T, 4> l)
      : array(nested_initializer_list_shape_(l)) {
    copy_initializer_list_(l);
  }

  //----------------------------------------------------------------------------

  auto clone() const {
    array tmp(shape_);
    // TODO: use GPU
    for (size_t i = 0; i < element_count(); i++) {
      tmp.at(i) = at(i);
    }
    return tmp;
  }

  //----------------------------------------------------------------------------

  auto operator==(const array &rhs) const {
    if (this != &rhs) {
      if (shape_ != rhs.shape_) {
        return false;
      }
      // TODO: use GPU
      for (size_t i = 0; i < element_count(); i++) {
        if (!equal_value_(at(i), rhs.at(i))) {
          return false;
        }
      }
    }
    return true;
  }

  //----------------------------------------------------------------------------

  auto buffer_element_count() const { return buf_len_; }

  auto buffer_bytes() const { return buf_len_ * sizeof(T); }

  auto buffer_data() { return static_cast<T *>(buf_->contents()) + buf_off_; }

  auto buffer_data() const {
    return static_cast<const T *>(buf_->contents()) + buf_off_;
  }

  //----------------------------------------------------------------------------

  auto element_count() const {
    // TODO: cache
    return element_count_from_shape_(shape_);
  }

  auto length() const {
    if (shape_.empty()) {
      throw std::runtime_error("array: cannot call with a scalar value.");
    }
    return shape_[0];
  }

  auto dimension() const { return shape_.size(); }

  const auto &shape() const { return shape_; }

  const auto &strides() const { return strides_; }

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

  auto at() const { return *buffer_data(); }

  auto &at() { return *buffer_data(); }

  auto at(size_t i) const {
    bounds_check_(i);
    return buffer_data()[i % buffer_element_count()];
  }

  auto &at(size_t i) {
    bounds_check_(i);
    return buffer_data()[i % buffer_element_count()];
  }

  auto at(size_t x, size_t y) const {
    bounds_check_(x, y);
    return buffer_data()[strides_[0] * x + y];
  }

  auto &at(size_t x, size_t y) {
    bounds_check_(x, y);
    return buffer_data()[strides_[0] * x + y];
  }

  auto at(size_t x, size_t y, size_t z) const {
    bounds_check_(x, y, z);
    return buffer_data()[(strides_[0] * x) + (strides_[1] * y) + z];
  }

  auto &at(size_t x, size_t y, size_t z) {
    bounds_check_(x, y, z);
    return buffer_data()[(strides_[0] * x) + (strides_[1] * y) + z];
  }

  operator T() const { return at(); }

  operator T &() { return at(); }

  template <size_t I>
  constexpr auto take() const {
    if constexpr (I == 0) {
      return std::tuple<>();
    } else {
      auto t = take<I - 1>();
      return std::tuple_cat(t, std::tuple<T>(at(I - 1)));
    }
  }

  //----------------------------------------------------------------------------

  auto operator()(size_t row) const {
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

    value_type operator*() { return (*arr_)(i_); }

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

    value_type operator*() const { return (*arr_)(i_); }

    friend bool operator==(const const_row_iterator &a,
                           const const_row_iterator &b) {
      return a.i_ == b.i_;
    };

   private:
    const array *arr_ = nullptr;
    size_t i_ = 0;
  };

  auto begin() { return row_iterator(this, 0); }
  auto end() { return row_iterator(this, shape_[0]); }

  auto cbegin() const { return const_row_iterator(this, 0); }
  auto cend() const { return const_row_iterator(this, shape_[0]); }

  //----------------------------------------------------------------------------

  template <size_t I>
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

    auto operator*() const { return (*arr_)(i_).template take<I>(); }

    friend bool operator==(const row_tuple_iterator &a,
                           const row_tuple_iterator &b) {
      return a.i_ == b.i_;
    };

   private:
    array *arr_ = nullptr;
    size_t i_ = 0;
  };

  template <size_t I>
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

    auto operator*() const { return (*arr_)(i_).template take<I>(); }

    friend bool operator==(const const_row_tuple_iterator &a,
                           const const_row_tuple_iterator &b) {
      return a.i_ == b.i_;
    };

   private:
    const array *arr_ = nullptr;
    size_t i_ = 0;
  };

  template <size_t I>
  struct row_tuple_range {
    row_tuple_range(array *arr) : arr_(arr) {}
    auto begin() { return row_tuple_iterator<I>(arr_, 0); }
    auto end() { return row_tuple_iterator<I>(arr_, arr_->shape()[0]); }
    array *arr_ = nullptr;
  };

  template <size_t I>
  struct const_row_tuple_range {
    const_row_tuple_range(array *arr) : arr_(arr) {}
    auto begin() const { return row_tuple_iterator<I>(arr_, 0); }
    auto end() const { return row_tuple_iterator<I>(arr_, arr_->shape()[0]); }
    const array *arr_ = nullptr;
  };

  template <size_t I>
  auto rows() {
    return row_tuple_range<I>(this);
  }

  template <size_t I>
  auto rows() const {
    return const_row_tuple_range<I>(this);
  }

  //----------------------------------------------------------------------------

  void set(std::ranges::input_range auto &&r) {
    std::ranges::copy(r, element_begin());
  }
  void set(std::initializer_list<T> l) {
    std::ranges::copy(l, element_begin());
  }

  //----------------------------------------------------------------------------

  friend auto operator+(const array &lhs, const array &rhs) {
    return binary_operation_(lhs, rhs, Operation::Add);
  }

  friend auto operator-(const array &lhs, const array &rhs) {
    return binary_operation_(lhs, rhs, Operation::Sub);
  }

  friend auto operator*(const array &lhs, const array &rhs) {
    return binary_operation_(lhs, rhs, Operation::Mul);
  }

  friend auto operator/(const array &lhs, const array &rhs) {
    return binary_operation_(lhs, rhs, Operation::Div);
  }

  //------------------------------------------------------------------------------

  friend auto operator+(const array &lhs, auto rhs) {
    return lhs + array(static_cast<T>(rhs));
  }

  friend auto operator+(auto lhs, const array &rhs) {
    return array(static_cast<T>(lhs)) + rhs;
  }

  friend auto operator-(const array &lhs, auto rhs) {
    return lhs - array(static_cast<T>(rhs));
  }

  friend auto operator-(auto lhs, const array &rhs) {
    return array(static_cast<T>(lhs)) - rhs;
  }

  friend auto operator*(const array &lhs, auto rhs) {
    return lhs * array(static_cast<T>(rhs));
  }

  friend auto operator*(auto lhs, const array &rhs) {
    return array(static_cast<T>(lhs)) * rhs;
  }

  friend auto operator/(const array &lhs, auto rhs) {
    return lhs / array(static_cast<T>(rhs));
  }

  friend auto operator/(auto lhs, const array &rhs) {
    return array(static_cast<T>(lhs)) / rhs;
  }

  //----------------------------------------------------------------------------

  void constants(T val) {
    // TODO: use GPU
    std::fill(buffer_data(), buffer_data() + buffer_element_count(), val);
  }

  void zeros() { constants(0); };

  void ones() { constants(1); };

  void random() {
    // TODO: use GPU
    std::generate(buffer_data(), buffer_data() + buffer_element_count(), []() {
      return static_cast<T>(static_cast<double>(rand()) / RAND_MAX);
    });
  }

  //----------------------------------------------------------------------------

  array dot(const array &rhs) const {
    // TODO: use GPU
    if (dimension() == 1 && rhs.dimension() == 1 &&
        shape_[0] == rhs.shape_[0]) {
      auto tmp = array(shape_type{});

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
      auto tmp = array(shape_type{rows, cols});

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
      auto tmp = array(shape_type{cols});

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
      auto tmp = array(shape_type{rows});

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

  auto transpose() const {
    // TODO: use GPU
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

  auto sum() const {
    return std::accumulate(element_cbegin(), element_cend(), 0);
  }

  template <typename U = float>
  auto mean() const {
    return std::accumulate(element_cbegin(), element_cend(), 0) /
           static_cast<U>(element_count());
  }

 private:
  auto element_count_from_shape_(const shape_type &shape) const {
    size_t count = 1;
    for (auto n : shape) {
      count *= n;
    }
    return count;
  }

  //----------------------------------------------------------------------------

  auto equal_value_(float a, float b) const {
    // NOTE: is `1e-3` too large?
    return std::abs(a - b) < 1e-3;
  }

  auto equal_value_(int a, int b) const { return a == b; }

  //----------------------------------------------------------------------------

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
      auto tmp = array(lhs.shape());
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
        auto tmp = array(lhs.shape());
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

//------------------------------------------------------------------------------

template <typename T>
inline auto print(std::ostream &os, const array<T> &arr, size_t dim,
                    size_t arr_index) {
  if (dim + 1 == arr.dimension()) {
    for (size_t i = 0; i < arr.shape()[dim]; i++, arr_index++) {
      if (i > 0) {
        os << ' ';
      }
      os << arr.at(arr_index);
    }
    return arr_index;
  }

  for (size_t i = 0; i < arr.shape()[dim]; i++) {
    // if (dim == 0 && i > 0) {
    if (dim < arr.dimension() && i > 0) {
      os << "\n ";
      for (size_t j = 0; j < dim; j++) {
        os << ' ';
      }
    }
    os << '[';
    arr_index = print(os, arr, dim + 1, arr_index);
    os << ']';
  }
  return arr_index;
}

template <typename T>
inline auto &operator<<(std::ostream &os, const array<T> &arr) {
  if (arr.dimension() == 0) {
    os << arr.at();
  } else {
    os << "[";
    print(os, arr, 0, 0);
    os << "]";
  }
  return os;
}

//------------------------------------------------------------------------------

template <typename T>
inline auto vector(size_t length) {
  return array<T>(shape_type{length});
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
  return array<T>(shape_type{row, col});
}

template <typename T>
inline auto matrix(size_t row, size_t col, std::ranges::input_range auto &&r) {
  auto tmp = matrix<T>(row, col);
  tmp.set(r);
  return tmp;
}

template <typename T>
inline auto matrix(nested_initializer_list<T, 2> l) {
  return array<T>(l);
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
