#pragma once

#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <CoreFoundation/CoreFoundation.h>

namespace sil {
namespace objc {

struct mtl_size {
  unsigned long width, height, depth;
};

// Cached selectors
inline SEL sel(const char* name) {
  return sel_registerName(name);
}

// Basic message sends
inline void* send(void* obj, const char* sel_name) {
  return reinterpret_cast<void*(*)(void*, SEL)>(objc_msgSend)(
      obj, sel(sel_name));
}

inline void* send(void* obj, SEL s) {
  return reinterpret_cast<void*(*)(void*, SEL)>(objc_msgSend)(obj, s);
}

inline void* send(void* obj, const char* sel_name, void* a) {
  return reinterpret_cast<void*(*)(void*, SEL, void*)>(objc_msgSend)(
      obj, sel(sel_name), a);
}

inline void* send(void* obj, const char* sel_name, void* a, size_t b, size_t c) {
  return reinterpret_cast<void*(*)(void*, SEL, void*, size_t, size_t)>(
      objc_msgSend)(obj, sel(sel_name), a, b, c);
}

inline void* send(void* obj, const char* sel_name, const void* a, size_t b, size_t c) {
  return reinterpret_cast<void*(*)(void*, SEL, const void*, size_t, size_t)>(
      objc_msgSend)(obj, sel(sel_name), a, b, c);
}

inline void* send(void* obj, const char* sel_name, size_t a, size_t b) {
  return reinterpret_cast<void*(*)(void*, SEL, size_t, size_t)>(
      objc_msgSend)(obj, sel(sel_name), a, b);
}

inline size_t send_uint(void* obj, const char* sel_name) {
  return reinterpret_cast<size_t(*)(void*, SEL)>(objc_msgSend)(
      obj, sel(sel_name));
}

inline void send_dispatch(void* obj, mtl_size grid, mtl_size threads) {
  reinterpret_cast<void(*)(void*, SEL, mtl_size, mtl_size)>(objc_msgSend)(
      obj, sel("dispatchThreads:threadsPerThreadgroup:"), grid, threads);
}

// MPS-specific sends
inline void* send_mps_desc(void* cls, size_t rows, size_t cols,
                           size_t row_bytes, unsigned long dtype) {
  return reinterpret_cast<void*(*)(void*, SEL, size_t, size_t, size_t,
                                   unsigned long)>(objc_msgSend)(
      cls,
      sel("matrixDescriptorWithRows:columns:rowBytes:dataType:"),
      rows, cols, row_bytes, dtype);
}

inline void* send_mps_matmul_init(void* obj, void* device,
                                   bool tl, bool tr,
                                   size_t rows, size_t cols, size_t inner,
                                   double alpha, double beta) {
  return reinterpret_cast<void*(*)(void*, SEL, void*, bool, bool,
                                   size_t, size_t, size_t,
                                   double, double)>(objc_msgSend)(
      obj,
      sel("initWithDevice:transposeLeft:transposeRight:resultRows:"
          "resultColumns:interiorColumns:alpha:beta:"),
      device, tl, tr, rows, cols, inner, alpha, beta);
}

inline void send_mps_encode(void* matmul, void* cb, void* left, void* right,
                            void* result) {
  reinterpret_cast<void(*)(void*, SEL, void*, void*, void*, void*)>(
      objc_msgSend)(
      matmul,
      sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"),
      cb, left, right, result);
}

// NSString / CFString creation
inline void* cfstr(const char* s) {
  return (void*)CFStringCreateWithCString(nullptr, s, kCFStringEncodingUTF8);
}

inline void cfrelease(void* ref) {
  CFRelease((CFTypeRef)ref);
}

inline void* cls(const char* name) {
  return (void*)objc_getClass(name);
}

inline void release(void* obj) {
  send(obj, "release");
}

}  // namespace objc
};  // namespace sil
