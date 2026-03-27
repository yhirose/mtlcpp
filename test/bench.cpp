#include <sil.h>

#include <eigen3/Eigen/Core>

#include <mlx/mlx.h>
namespace mx = mlx::core;

#include <chrono>

struct BenchEntry {
  std::string name;
  double seconds;
};

using BenchGroup = std::pair<std::string, std::vector<BenchEntry>>;

double measure(size_t iters, auto fn) {
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

void print_bar_graphs(const std::vector<BenchGroup>& groups) {
  constexpr int bar_width = 50;

  std::puts("");

  // Find max name width across all groups
  constexpr int name_width = 5;  // "Auto " "AMX  " "MPS  " "Eigen" "MLX  "

  auto section_name = [](const std::string& title) {
    auto pos = title.find(" (");
    return pos != std::string::npos ? title.substr(0, pos) : title;
  };

  std::string current_section;
  for (auto& [title, entries] : groups) {
    auto sec = section_name(title);
    if (sec != current_section) {
      current_section = sec;
      std::printf("── %s ──────────────────────────────────────────\n\n", sec.c_str());
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

      std::printf("    %-*s %s %s %6.1fx\n",
                  name_width, e.name.c_str(),
                  bar.c_str(), time_str.c_str(), ratio);
    }
    std::puts("");
  }
}

constexpr size_t fast_iters = 1000;
constexpr size_t slow_iters = 200;

void bench_mtl(std::vector<BenchEntry>& entries,
               size_t iters, auto setup_and_run) {
  for (auto [mode, name] : {std::pair{sil::Device::Auto, "Auto"},
                             std::pair{sil::Device::CPU, "CPU"}}) {
    switch (mode) {
      case sil::Device::CPU: sil::use_cpu(); break;
      case sil::Device::Auto: sil::use_auto(); break;
      default: break;
    }
    auto t = setup_and_run(name, iters);
    entries.push_back({name, t});
  }
}

void add(std::vector<BenchGroup>& groups) {
  for (auto n : {100'000ul, 1'000'000ul, 10'000'000ul}) {
    size_t iters = n <= 100'000 ? fast_iters : n <= 1'000'000 ? 200 : 50;

    std::vector<BenchEntry> entries;

    bench_mtl(entries, iters, [&](auto& mode, auto iters) {
      auto a = sil::ones<float>({n});
      auto b = sil::ones<float>({n});
      auto c = sil::array<float>();
      return measure(iters, [&] { c = a + b; });
    });

    {
      auto a_s = sil::storage::make(n * sizeof(float));
      auto b_s = sil::storage::make(n * sizeof(float));
      auto out_s = sil::storage::make(n * sizeof(float));
      a_s.len = n;
      b_s.len = n;
      out_s.len = n;
      std::fill_n(static_cast<float*>(a_s.data), n, 1.0f);
      std::fill_n(static_cast<float*>(b_s.data), n, 1.0f);
      entries.push_back({"MSL", measure(iters, [&] { sil::msl::add<float>(a_s, b_s, out_s); })});
    }

    {
      Eigen::VectorXf aa = Eigen::VectorXf::Ones(n);
      Eigen::VectorXf bb = Eigen::VectorXf::Ones(n);
      Eigen::VectorXf cc(n);
      entries.push_back({"Eigen", measure(iters, [&] { cc = aa + bb; })});
    }

    {
      auto ma = mx::ones({static_cast<int>(n)});
      auto mb = mx::ones({static_cast<int>(n)});
      mx::eval(ma, mb);
      auto mc = mx::array(0.0f);
      entries.push_back({"MLX", measure(iters, [&] { mc = mx::add(ma, mb); mx::eval(mc); })});
    }

    groups.push_back({std::format("add ({})", n), std::move(entries)});
  }
}

void add_scalar(std::vector<BenchGroup>& groups) {
  for (auto n : {100'000ul, 1'000'000ul, 10'000'000ul}) {
    size_t iters = n <= 100'000 ? fast_iters : n <= 1'000'000 ? 200 : 50;

    std::vector<BenchEntry> entries;

    bench_mtl(entries, iters, [&](auto& mode, auto iters) {
      auto a = sil::ones<float>({n});
      auto s = sil::array<float>({1.0f});
      auto c = sil::array<float>();
      return measure(iters, [&] { c = a + s; });
    });

    {
      auto a_s = sil::storage::make(n * sizeof(float));
      auto b_s = sil::storage::make(1 * sizeof(float));
      auto out_s = sil::storage::make(n * sizeof(float));
      a_s.len = n;
      b_s.len = 1;
      out_s.len = n;
      std::fill_n(static_cast<float*>(a_s.data), n, 1.0f);
      *static_cast<float*>(b_s.data) = 1.0f;
      entries.push_back({"MSL", measure(iters, [&] { sil::msl::add<float>(a_s, b_s, out_s); })});
    }

    {
      Eigen::VectorXf aa = Eigen::VectorXf::Ones(n);
      Eigen::VectorXf cc(n);
      entries.push_back({"Eigen", measure(iters, [&] { cc = aa.array() + 1.0f; })});
    }

    {
      auto ma = mx::ones({static_cast<int>(n)});
      mx::eval(ma);
      auto ms = mx::array(1.0f);
      auto mc = mx::array(0.0f);
      entries.push_back({"MLX", measure(iters, [&] { mc = mx::add(ma, ms); mx::eval(mc); })});
    }

    groups.push_back({std::format("add scalar ({})", n), std::move(entries)});
  }
}

void dot(std::vector<BenchGroup>& groups) {
  struct DotSize { size_t m; size_t fast; size_t slow; };
  for (auto [m, fast, slow] : std::initializer_list<DotSize>{
           {64, 3000, 200}, {1024, 30, 20}, {4096, 10, 10}}) {
    std::vector<BenchEntry> entries;

    // Auto
    sil::use_auto();
    {
      auto a = sil::ones<float>({m, m});
      auto b = sil::ones<float>({m, m});
      auto c = sil::array<float>();
      entries.push_back({"Auto", measure(slow, [&] { c = a.dot(b); })});
    }

    // AMX
    sil::use_cpu();
    {
      auto a = sil::ones<float>({m, m});
      auto b = sil::ones<float>({m, m});
      auto c = sil::array<float>();
      entries.push_back({"CPU", measure(fast, [&] { c = a.dot(b); })});
    }

    // MPS
    sil::use_mps();
    {
      auto a = sil::ones<float>({m, m});
      auto b = sil::ones<float>({m, m});
      auto c = sil::array<float>();
      entries.push_back({"MPS", measure(slow, [&] { c = a.dot(b); })});
    }

    if (m <= 1024) {
      Eigen::MatrixXf aa = Eigen::MatrixXf::Ones(m, m);
      Eigen::MatrixXf bb = Eigen::MatrixXf::Ones(m, m);
      Eigen::MatrixXf cc;
      entries.push_back({"Eigen", measure(slow, [&] { cc = aa * bb; })});
    }

    {
      int mi = static_cast<int>(m);
      auto ma = mx::ones({mi, mi});
      auto mb = mx::ones({mi, mi});
      mx::eval(ma, mb);
      auto mc = mx::array(0.0f);
      entries.push_back({"MLX", measure(slow, [&] { mc = mx::matmul(ma, mb); mx::eval(mc); })});
    }

    groups.push_back({std::format("dot ({}x{})", m, m), std::move(entries)});
  }
}

int main(void) {
  std::vector<BenchGroup> groups;
  add(groups);
  add_scalar(groups);
  dot(groups);
  print_bar_graphs(groups);
}
