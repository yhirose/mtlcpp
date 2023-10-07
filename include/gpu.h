#pragma once

#include <common.h>

#include <cstdlib>
#include <memory>

namespace mtlcpp {

struct GPU {
  class MemoryImpl;

  class Memory {
   private:
    friend class GPU;

    std::shared_ptr<MemoryImpl> impl_;

   public:
    Memory();

    void *data();

    template <typename T>
    T *data() {
      return static_cast<T *>(data());
    }

    const void *data() const;

    template <typename T>
    const T *data() const {
      return static_cast<const T *>(data());
    }

    size_t length() const;

    template <typename T>
    size_t length() const {
      return length() / sizeof(T);
    }
  };

  static Memory allocate(size_t buf_len);

  template <typename T>
  static Memory allocate(size_t arr_len) {
    return allocate(sizeof(T) * arr_len);
  }

  static void compute(const Memory &A, const Memory &B, const Memory &OUT,
                      ComputeType id, size_t element_size);

  template <typename T>
  static void compute(const Memory &A, const Memory &B, const Memory &OUT,
                      ComputeType id) {
    compute(A, B, OUT, id, sizeof(T));
  }
};

};  // namespace mtlcpp

