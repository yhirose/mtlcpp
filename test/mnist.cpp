#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <array.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <map>

class memory_mapped_file {
 public:
  memory_mapped_file(const char* path) {
    fd_ = open(path, O_RDONLY);
    if (fd_ == -1) {
      std::runtime_error("");
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
      cleanup();
      std::runtime_error("");
    }
    size_ = sb.st_size;

    addr_ = ::mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (addr_ == MAP_FAILED) {
      cleanup();
      std::runtime_error("");
    }
  }

  ~memory_mapped_file() { cleanup(); }

  bool is_open() const { return addr_ != MAP_FAILED; }
  size_t size() const { return size_; }
  const char* data() const { return (const char*)addr_; }

 private:
  void cleanup() {
    if (addr_ != MAP_FAILED) {
      munmap(addr_, size_);
      addr_ = MAP_FAILED;
    }
    if (fd_ != -1) {
      close(fd_);
      fd_ = -1;
    }
    size_ = 0;
  }

  int fd_ = -1;
  size_t size_ = 0;
  void* addr_ = MAP_FAILED;
};

class mnist_data {
 public:
  mnist_data(const char* label_data_path, const char* image_data_path)
      : label_data_(label_data_path), image_data_(image_data_path) {
    if (read32(label_data_, 0) != 2049) {
      throw std::runtime_error("invalid label data format.");
    }

    if (read32(image_data_, 0) != 2051) {
      throw std::runtime_error("invalid image data format.");
    }

    number_of_items_ = read32(label_data_, 4);
    if (number_of_items_ != read32(image_data_, 4)) {
      throw std::runtime_error("image data doesn't match label data.");
    }

    labels_ = reinterpret_cast<const uint8_t*>(label_data_.data() + 8);

    number_of_rows_ = read32(image_data_, 8);
    number_of_columns_ = read32(image_data_, 12);
    pixels_ = reinterpret_cast<const uint8_t*>(image_data_.data() + 16);
  }

  size_t size() const { return number_of_items_; }

  // Label
  const uint8_t* label_data() const { return labels_; }

  size_t label(size_t i) const { return labels_[i]; }

  // Image
  const uint8_t* image_data() const { return pixels_; }

  const float* normalized_image_data() const {
    if (normalized_pixels_.empty()) {
      auto total_pixels = image_pixel_size() * size();
      normalized_pixels_.resize(total_pixels);
      for (size_t i = 0; i < total_pixels; i++) {
        normalized_pixels_[i] = (float)pixels_[i] / (float)255;
      }
    }
    return normalized_pixels_.data();
  }

  size_t image_rows() const { return number_of_rows_; }

  size_t image_columns() const { return number_of_columns_; }

  size_t image_pixel_size() const {
    return number_of_rows_ * number_of_columns_;
  }

  uint8_t pixel(size_t row, size_t col) const {
    return pixels_[number_of_columns_ * row + col];
  }

 private:
  uint32_t read32(const memory_mapped_file& mm, size_t off) {
    auto p = mm.data() + off;
    return __builtin_bswap32(*reinterpret_cast<const uint32_t*>(p));
  }

  size_t number_of_items_ = 0;
  size_t number_of_rows_ = 0;
  size_t number_of_columns_ = 0;

  const uint8_t* labels_ = nullptr;
  const uint8_t* pixels_ = nullptr;

  memory_mapped_file label_data_;
  memory_mapped_file image_data_;

  mutable std::vector<float> normalized_pixels_;
};

struct Network {
  std::map<std::string, mtl::array<float>> w;
  std::map<std::string, mtl::array<float>> b;
};

Network init_network() {
  auto split = [](const char* b, const char* e, char d,
                  std::function<void(const char*, const char*)> fn) {
    size_t i = 0;
    size_t beg = 0;
    while (e ? (b + i < e) : (b[i] != '\0')) {
      if (b[i] == d) {
        fn(&b[beg], &b[i]);
        beg = i + 1;
      }
      i++;
    }
    if (i) {
      fn(&b[beg], &b[i]);
    }
  };

  Network network;
  std::ifstream f("sample_weight.csv");
  std::string line;
  while (std::getline(f, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream s(line);

    std::string label;
    size_t rows;
    size_t cols;
    s >> label >> rows >> cols;

    std::vector<float> values;
    {
      values.reserve(rows * cols);

      auto count = rows;
      while (count > 0) {
        std::getline(f, line);
        split(&line[0], &line[line.size() - 1], ',',
              [&](auto b, auto /*e*/) { values.push_back(std::atof(b)); });
        count--;
      }
    }

    if (rows > 1) {
      network.w[label] = mtl::array<float>({rows, cols}, values);
    } else {
      network.b[label] = mtl::array<float>({cols}, values);
    }
  }
  return network;
}

auto predict(const Network& network, const mtl::array<float>& x) {
  auto W1 = network.w.at("W1");
  auto W2 = network.w.at("W2");
  auto W3 = network.w.at("W3");
  auto b1 = network.b.at("b1");
  auto b2 = network.b.at("b2");
  auto b3 = network.b.at("b3");

  auto a1 = x.dot(W1) + b1;
  auto z1 = a1.sigmoid();
  auto a2 = z1.dot(W2) + b2;
  auto z2 = a2.sigmoid();
  auto a3 = z2.dot(W3) + b3;
  auto y = a3.softmax();
  return y;
}

int main(void) {
  try {
    auto data = mnist_data("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte");
    auto network = init_network();

    // auto p = data.image_data();
    auto p = data.normalized_image_data();

    size_t batch_size = 100;
    size_t accuracy_cnt = 0;

    for (auto i = 0u; i < data.size(); i += batch_size) {
      auto x = mtl::array<float>({batch_size, data.image_pixel_size()},
                                 p + data.image_pixel_size() * i);
      auto y = predict(network, x);
      auto e = mtl::array<int>({batch_size}, data.label_data() + i);
      auto a = y.argmax();

      auto r = e == a;
      accuracy_cnt += r.count();
    }

    auto accuracy = (double)accuracy_cnt / (double)data.size();
    std::cout << "MNIST Accuracy: " << accuracy << std::endl;
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
  }
}
