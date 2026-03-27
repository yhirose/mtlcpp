#pragma once

#include <types.h>
#include <device.h>

#include <Accelerate/Accelerate.h>
#include <cmath>

namespace sil {

class cpu {
 public:
  template <value_type T>
  static void add(const storage &A, const storage &B,
                  storage &OUT);

  template <value_type T>
  static void sub(const storage &A, const storage &B,
                  storage &OUT);

  template <value_type T>
  static void mul(const storage &A, const storage &B,
                  storage &OUT);

  template <value_type T>
  static void div(const storage &A, const storage &B,
                  storage &OUT);

  template <value_type T>
  static void pow(const storage &A, const storage &B,
                  storage &OUT);

  template <value_type T>
  static void dot(const storage &A, const storage &B,
                  storage &OUT, uint32_t A_cols, uint32_t OUT_rows,
                  uint32_t OUT_cols);

 private:
  template <value_type T>
  static const T *ptr(const storage &s) {
    return static_cast<const T *>(s.data) + s.off;
  }

  template <value_type T>
  static T *mutable_ptr(storage &s) {
    return static_cast<T *>(s.data) + s.off;
  }

};

//-----------------------------------------------------------------------------
// Implementation
//-----------------------------------------------------------------------------

template <value_type T>
inline void cpu::add(const storage &A, const storage &B,
                      storage &OUT) {
  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  auto n = OUT.len;

  if constexpr (std::is_same_v<T, float>) {
    if (A.len == n && B.len == n) {
      vDSP_vadd(a, 1, b, 1, out, 1, n);
      return;
    }
    if (B.len == 1) { vDSP_vsadd(a, 1, b, out, 1, n); return; }
    if (A.len == 1) { vDSP_vsadd(b, 1, a, out, 1, n); return; }
    // Broadcast: repeat shorter side row-by-row with vDSP
    if (A.len == n && B.len > 1 && n % B.len == 0) {
      for (size_t i = 0; i < n; i += B.len)
        vDSP_vadd(a + i, 1, b, 1, out + i, 1, B.len);
      return;
    }
    if (B.len == n && A.len > 1 && n % A.len == 0) {
      for (size_t i = 0; i < n; i += A.len)
        vDSP_vadd(a, 1, b + i, 1, out + i, 1, A.len);
      return;
    }
  }
  if (A.len == n && B.len == n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
  } else {
    for (size_t i = 0; i < n; i++) out[i] = a[i % A.len] + b[i % B.len];
  }
}

template <value_type T>
inline void cpu::sub(const storage &A, const storage &B,
                      storage &OUT) {
  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  auto n = OUT.len;

  if constexpr (std::is_same_v<T, float>) {
    if (A.len == n && B.len == n) {
      vDSP_vsub(b, 1, a, 1, out, 1, n);
      return;
    }
    if (B.len == 1) {
      float neg_b = -b[0];
      vDSP_vsadd(a, 1, &neg_b, out, 1, n);
      return;
    }
    if (A.len == n && B.len > 1 && n % B.len == 0) {
      for (size_t i = 0; i < n; i += B.len)
        vDSP_vsub(b, 1, a + i, 1, out + i, 1, B.len);
      return;
    }
    if (B.len == n && A.len > 1 && n % A.len == 0) {
      for (size_t i = 0; i < n; i += A.len)
        vDSP_vsub(b + i, 1, a, 1, out + i, 1, A.len);
      return;
    }
  }
  if (A.len == n && B.len == n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] - b[i];
  } else {
    for (size_t i = 0; i < n; i++) out[i] = a[i % A.len] - b[i % B.len];
  }
}

template <value_type T>
inline void cpu::mul(const storage &A, const storage &B,
                      storage &OUT) {
  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  auto n = OUT.len;

  if constexpr (std::is_same_v<T, float>) {
    if (A.len == n && B.len == n) {
      vDSP_vmul(a, 1, b, 1, out, 1, n);
      return;
    }
    if (B.len == 1) { vDSP_vsmul(a, 1, b, out, 1, n); return; }
    if (A.len == 1) { vDSP_vsmul(b, 1, a, out, 1, n); return; }
    if (A.len == n && B.len > 1 && n % B.len == 0) {
      for (size_t i = 0; i < n; i += B.len)
        vDSP_vmul(a + i, 1, b, 1, out + i, 1, B.len);
      return;
    }
    if (B.len == n && A.len > 1 && n % A.len == 0) {
      for (size_t i = 0; i < n; i += A.len)
        vDSP_vmul(a, 1, b + i, 1, out + i, 1, A.len);
      return;
    }
  }
  if (A.len == n && B.len == n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
  } else {
    for (size_t i = 0; i < n; i++) out[i] = a[i % A.len] * b[i % B.len];
  }
}

template <value_type T>
inline void cpu::div(const storage &A, const storage &B,
                      storage &OUT) {
  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  auto n = OUT.len;

  if constexpr (std::is_same_v<T, float>) {
    if (A.len == n && B.len == n) {
      vDSP_vdiv(b, 1, a, 1, out, 1, n);
      return;
    }
    if (B.len == 1) { vDSP_vsdiv(a, 1, b, out, 1, n); return; }
    if (A.len == 1) { vDSP_svdiv(a, b, 1, out, 1, n); return; }
    if (A.len == n && B.len > 1 && n % B.len == 0) {
      for (size_t i = 0; i < n; i += B.len)
        vDSP_vdiv(b, 1, a + i, 1, out + i, 1, B.len);
      return;
    }
    if (B.len == n && A.len > 1 && n % A.len == 0) {
      for (size_t i = 0; i < n; i += A.len)
        vDSP_vdiv(b + i, 1, a, 1, out + i, 1, A.len);
      return;
    }
  }
  if (A.len == n && B.len == n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] / b[i];
  } else {
    for (size_t i = 0; i < n; i++) out[i] = a[i % A.len] / b[i % B.len];
  }
}

template <value_type T>
inline void cpu::pow(const storage &A, const storage &B,
                      storage &OUT) {
  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  auto n = OUT.len;

  if constexpr (std::is_same_v<T, float>) {
    if (A.len == n && B.len == n) {
      int n_int = static_cast<int>(n);
      vvpowf(out, b, a, &n_int);
      return;
    }
  }
  if (A.len == n && B.len == n) {
    for (size_t i = 0; i < n; i++) out[i] = std::pow(a[i], b[i]);
  } else {
    for (size_t i = 0; i < n; i++)
      out[i] = std::pow(a[i % A.len], b[i % B.len]);
  }
}

template <value_type T>
inline void cpu::dot(const storage &A, const storage &B,
                      storage &OUT, uint32_t A_cols, uint32_t OUT_rows,
                      uint32_t OUT_cols) {
  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, OUT_rows, OUT_cols,
                A_cols, 1.0f, ptr<T>(A), A_cols, ptr<T>(B), OUT_cols, 0.0f,
                mutable_ptr<T>(OUT), OUT_cols);
    return;
  }

  const auto *a = ptr<T>(A);
  const auto *b = ptr<T>(B);
  auto *out = mutable_ptr<T>(OUT);
  std::memset(out, 0, OUT_rows * OUT_cols * sizeof(T));

  // i,k,j loop order for cache-friendly access to B
  for (uint32_t i = 0; i < OUT_rows; i++) {
    for (uint32_t k = 0; k < A_cols; k++) {
      auto a_ik = a[i * A_cols + k];
      for (uint32_t j = 0; j < OUT_cols; j++) {
        out[i * OUT_cols + j] += a_ik * b[k * OUT_cols + j];
      }
    }
  }
}

};  // namespace sil
