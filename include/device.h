#pragma once

#include <unified_memory.h>

#include <chrono>
#include <limits>
#include <unordered_map>
#include <vector>

namespace sil {

enum class Device {
  Auto = 0,
  MPS,
  CPU,
};

inline Device device_ = Device::Auto;

inline void use_cpu() { device_ = Device::CPU; }
inline void use_mps() { device_ = Device::MPS; }
inline void use_auto() { device_ = Device::Auto; }

//-----------------------------------------------------------------------------

class device_cache {
  std::unordered_map<uint64_t, Device> cache_;

  device_cache() = default;

 public:
  device_cache(const device_cache &) = delete;
  device_cache &operator=(const device_cache &) = delete;

  static device_cache &instance() {
    static device_cache c;
    return c;
  }

  static size_t bucket(size_t n) {
    size_t b = 1;
    while (b < n) b <<= 1;
    return b;
  }

  static uint64_t key(uint32_t op, size_t b) {
    return (static_cast<uint64_t>(op) << 48) | b;
  }

  bool lookup(uint64_t k, Device &out) const {
    auto it = cache_.find(k);
    if (it == cache_.end()) return false;
    out = it->second;
    return true;
  }

  void store(uint64_t k, Device d) { cache_[k] = d; }

  // Measure execution time of fn and return best-of-3 elapsed seconds
  static double time(auto fn) {
    fn();  // warmup
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      fn();
      auto t1 = std::chrono::high_resolution_clock::now();
      best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
  }
};

};  // namespace sil
