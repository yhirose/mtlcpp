#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <format>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <mach/mach.h>

struct BenchEntry {
  std::string name;
  double seconds;
};

using BenchGroup = std::pair<std::string, std::vector<BenchEntry>>;

inline double measure(size_t iters, auto fn) {
  // Warmup
  for (size_t i = 0; i < std::min(iters, size_t(10)); i++) fn();

  double best = std::numeric_limits<double>::max();
  for (size_t i = 0; i < iters; i++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }
  return best;
}

inline double gflops_gemm(size_t M, size_t N, size_t K, double seconds) {
  return 2.0 * M * N * K / seconds / 1e9;
}

inline size_t peak_rss_bytes() {
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
  return info.resident_size;
}

inline void print_group(const BenchGroup& group, int name_width = 0) {
  constexpr int bar_width = 50;

  auto& [title, entries] = group;

  if (name_width == 0) {
    name_width = 5;
    for (auto& e : entries)
      name_width = std::max(name_width, static_cast<int>(e.name.size()));
  }

  std::printf("  %s\n", title.c_str());

  double fastest =
      std::ranges::min_element(entries, {}, &BenchEntry::seconds)->seconds;

  for (auto& e : entries) {
    double ratio = e.seconds / fastest;
    int bars = std::max(1, static_cast<int>(std::round(bar_width / ratio)));

    std::string bar;
    for (int i = 0; i < bars; ++i) bar += "\xe2\x96\x85";
    for (int i = bars; i < bar_width; ++i) bar += ' ';

    auto time_str = e.seconds < 1e-3
                        ? std::format("{:>9.2f} us", e.seconds * 1e6)
                        : std::format("{:>9.2f} ms", e.seconds * 1e3);

    std::printf("    %-*s %s %s %6.1fx\n", name_width, e.name.c_str(),
                bar.c_str(), time_str.c_str(), ratio);
  }
  std::puts("");
  std::fflush(stdout);
}

inline void print_section(const char* name) {
  std::printf("\n─ %s ─────────────────────────────────────\n\n", name);
  std::fflush(stdout);
}

inline void print_bar_graphs(const std::vector<BenchGroup>& groups) {
  int name_width = 5;
  for (auto& [title, entries] : groups)
    for (auto& e : entries)
      name_width = std::max(name_width, static_cast<int>(e.name.size()));

  auto section_name = [](const std::string& title) {
    auto pos = title.find(" (");
    return pos != std::string::npos ? title.substr(0, pos) : title;
  };

  std::string current_section;
  for (auto& group : groups) {
    auto sec = section_name(group.first);
    if (sec != current_section) {
      current_section = sec;
      print_section(sec.c_str());
    }
    print_group(group, name_width);
  }
}

inline void print_csv(const std::vector<BenchGroup>& groups) {
  std::printf("benchmark,library,best_us\n");
  for (auto& [title, entries] : groups) {
    for (auto& e : entries) {
      std::printf("%s,%s,%.2f\n", title.c_str(), e.name.c_str(),
                  e.seconds * 1e6);
    }
  }
}

inline bool has_csv_flag(int argc, const char** argv) {
  for (int i = 1; i < argc; i++) {
    if (std::string("--csv") == argv[i]) return true;
  }
  return false;
}
