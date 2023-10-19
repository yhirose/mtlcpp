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
  size_t length_ = 0;
  mtl::managed_ptr<MTL::Buffer> buf_;

 public:
  array(array &&rhs) = default;
  array(const array &rhs) = default;             // TODO: use GPU
  array &operator=(const array &rhs) = default;  // TODO: use GPU

  array(const shape_type &shape) {
    reshape(shape);
    allocate_buffer_();
  }

  array(T val) : array(shape_type({})) { *data() = val; }

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

  array clone() const {
    array tmp(shape_);
    // TODO: use GPU
    tmp.set(cbegin(), cend());
    return tmp;
  }

  //----------------------------------------------------------------------------

  bool operator==(const array &rhs) const {
    if (this != &rhs) {
      if (shape_ != rhs.shape_) {
        return false;
      }
      // TODO: use GPU
      for (size_t i = 0; i < length(); i++) {
        if (at(i) != rhs.at(i)) {
          return false;
        }
      }
    }
    return true;
  }

  //----------------------------------------------------------------------------

  size_t length() const { return length_; }

  size_t buffer_length() const { return buf_->length() / sizeof(T); }

  size_t buffer_bytes() const { return buf_->length(); }

  //----------------------------------------------------------------------------

  const shape_type &shape() const { return shape_; }

  const strides_type &strides() const { return strides_; }

  size_t shape(size_t i) const {
    if (i >= shape_.size()) {
      throw std::runtime_error("array: index is out of bounds.");
    }
    return shape_[i];
  }

  size_t dimension() const { return shape_.size(); }

  void reshape(const shape_type &shape) {
    // TODO: check the shape

    shape_ = shape;

    // length
    length_ = 1;
    for (auto n : shape) {
      length_ *= n;
    }

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
      if (shape(i) != target_shape[i + diff]) {
        throw std::runtime_error("array: invalid shape for broadcast.");
      }
    }

    array tmp = *this;
    tmp.shape_ = target_shape;

    // length
    tmp.length_ = 1;
    for (auto n : target_shape) {
      tmp.length_ *= n;
    }

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

  T *data() { return static_cast<T *>(buf_->contents()); }

  const T *data() const { return static_cast<const T *>(buf_->contents()); }

  //----------------------------------------------------------------------------

  T at() const { return *data(); }

  T &at() { return *data(); }

  T at(size_t i) const {
    bounds_check_(i);
    return data()[i % buffer_length()];
  }

  T &at(size_t i) {
    bounds_check_(i);
    return data()[i % buffer_length()];
  }

  T at(size_t x, size_t y) const {
    bounds_check_(x, y);
    return data()[strides_[0] * x + y];
  }

  T &at(size_t x, size_t y) {
    bounds_check_(x, y);
    return data()[strides_[0] * x + y];
  }

  T at(size_t x, size_t y, size_t z) const {
    bounds_check_(x, y, z);
    return data()[(strides_[0] * x) + (strides_[1] * y) + z];
  }

  T &at(size_t x, size_t y, size_t z) {
    bounds_check_(x, y, z);
    return data()[(strides_[0] * x) + (strides_[1] * y) + z];
  }

  //----------------------------------------------------------------------------

  T operator[](size_t i) const { return at(i); }

  T &operator[](size_t i) { return at(i); }

  //----------------------------------------------------------------------------

  class iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using reference = T &;
    using iterator_concept = std::forward_iterator_tag;

    iterator(array *arr, size_t i) : arr_(arr), i_(i) {}

    iterator &operator++() {
      ++i_;
      return *this;
    }

    iterator operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    reference &operator*() { return arr_->at(i_); }

    friend bool operator==(const iterator &a, const iterator &b) {
      return a.i_ == b.i_;
    };

   private:
    array *arr_ = nullptr;
    size_t i_ = 0;
  };

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, length()); }

  class const_iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using iterator_concept = std::forward_iterator_tag;

    const_iterator(const array *arr, size_t i) : arr_(arr), i_(i) {}

    const_iterator &operator++() {
      ++i_;
      return *this;
    }

    const_iterator operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    value_type operator*() const { return arr_->at(i_); }

    friend bool operator==(const const_iterator &a, const const_iterator &b) {
      return a.i_ == b.i_;
    };

   private:
    const array *arr_ = nullptr;
    size_t i_ = 0;
  };

  const_iterator cbegin() const { return const_iterator(this, 0); }
  const_iterator cend() const { return const_iterator(this, length()); }

  //----------------------------------------------------------------------------

  void set(std::input_iterator auto b, std::input_iterator auto e) {
    std::copy(b, e, begin());
  }
  void set(std::ranges::input_range auto &&r) { std::ranges::copy(r, begin()); }
  void set(std::initializer_list<T> l) { std::ranges::copy(l, begin()); }

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

  array dot(const array &rhs) const {
    // TODO: use GPU
    if (dimension() == 1 && rhs.dimension() == 1 && shape(0) == rhs.shape(0)) {
      auto tmp = array(shape_type{});

      T val = 0;
      for (size_t i = 0; i < shape(0); i++) {
        val += at(i) * rhs.at(i);
      }
      tmp.at() = val;
      return tmp;
    }

    if (dimension() == 2 && rhs.dimension() == 2 && shape(1) == rhs.shape(0)) {
      auto rows = shape(0);
      auto cols = rhs.shape(1);
      auto tmp = array(shape_type{rows, cols});

      for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
          T val = 0;
          for (size_t i = 0; i < shape(1); i++) {
            val += at(row, i) * rhs.at(i, col);
          }
          tmp.at(row, col) = val;
        }
      }
      return tmp;
    }

    if (dimension() == 1 && rhs.dimension() == 2 && shape(0) == rhs.shape(0)) {
      auto rows = 1;
      auto cols = rhs.shape(1);
      auto tmp = array(shape_type{cols});

      for (size_t col = 0; col < cols; col++) {
        T val = 0;
        for (size_t i = 0; i < shape(0); i++) {
          val += at(i) * rhs.at(i, col);
        }
        tmp.at(col) = val;
      }
      return tmp;
    }

    if (dimension() == 2 && rhs.dimension() == 1 && shape(1) == rhs.shape(0)) {
      auto rows = shape(0);
      auto tmp = array(shape_type{rows});

      for (size_t row = 0; row < rows; row++) {
        T val = 0;
        for (size_t i = 0; i < shape(1); i++) {
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
      tmp.reshape({1, length()});

      auto it = cbegin();
      for (size_t col = 0; col < length(); col++) {
        tmp.at(0, col) = *it;
        ++it;
      }
      return tmp;
    }

    if (dimension() == 2) {
      if (shape_[0] == 1) {
        auto tmp = clone();
        tmp.reshape({length()});

        auto it = cbegin();
        for (size_t row = 0; row < length(); row++) {
          tmp.at(row, 0) = *it;
          ++it;
        }
        return tmp;
      } else {
        auto shape = shape_;
        std::ranges::reverse(shape);

        auto tmp = clone();
        tmp.reshape(shape);

        auto it = cbegin();
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

      auto it = cbegin();
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

  auto sum() const { return std::accumulate(cbegin(), cend(), 0); }

  template <typename U = float>
  auto mean() const {
    return std::accumulate(cbegin(), cend(), 0) / static_cast<U>(length());
  }

 private:
  void allocate_buffer_() {
    size_t len = 1;
    for (auto n : shape_) {
      len *= n;
    }
    buf_ = mtl::newBuffer(sizeof(T) * len);
  }

  void allocate_buffer_and_copy_(const auto &l) {
    allocate_buffer_();
    if (nested_initializer_item_count_(l) != length()) {
      throw std::runtime_error("array: invalid initializer list.");
    }
    nested_initializer_copy_(data(), l);
  }

  void copy_initializer_list_(const auto &l) {
    if (nested_initializer_item_count_(l) != length()) {
      throw std::runtime_error("array: invalid initializer list.");
    }
    nested_initializer_copy_(data(), l);
  }

  //----------------------------------------------------------------------------

  void bounds_check_(size_t i) const {
    if (strides_.empty() || i >= length()) {
      throw std::runtime_error("array: index is out of bounds.");
    }
  }

  void bounds_check_(size_t x, size_t y) const {
    // TODO: implement
  }

  void bounds_check_(size_t x, size_t y, size_t z) const {
    // TODO: implement
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
      mtl::compute<T>(lhs.buf_, rhs.buf_, tmp.buf_, ope);
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
        for (size_t i = 0; i < lhs.length(); i++) {
          tmp[i] = fn(lhs[i], rhs[i]);
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
inline size_t print(std::ostream &os, const array<T> &arr, size_t dim,
                    size_t arr_index) {
  if (dim + 1 == arr.dimension()) {
    for (size_t i = 0; i < arr.shape(dim); i++, arr_index++) {
      if (i > 0) {
        os << ' ';
      }
      os << arr.at(arr_index);
    }
    return arr_index;
  }

  for (size_t i = 0; i < arr.shape(dim); i++) {
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
inline std::ostream &operator<<(std::ostream &os, const array<T> &arr) {
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
