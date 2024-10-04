#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>

std::size_t calculatePadding(const char *address, std::size_t alignment) {
  return ((alignment - reinterpret_cast<std::uintptr_t>(address)) &
          (alignment - 1));
}

template <std::size_t N> class MonotonicBuffer {
public:
  char d_buffer[N]; // fixed-size buffer
  char *d_top_p;    // next available address
  MonotonicBuffer() : d_top_p{d_buffer} {}
  template <typename T> void *allocate() {
    auto padding{calculatePadding(d_top_p, alignof(T))};
    auto delta{padding + sizeof(T)};
    if (((d_buffer + N) - d_top_p) < delta) {
      // not enough properly aligned unused space remaining
      return 0;
    }
    auto alignedAddres{d_top_p + padding};
    d_top_p += delta;
    return alignedAddres;
  };
};

int main(int argc, char **argv) {
  auto mb{MonotonicBuffer<20>()};
  auto cp{static_cast<char *>(mb.allocate<char>())};
  auto dp{static_cast<double *>(mb.allocate<double>())};
  for (decltype(1.00e+2F) i = 0; i < 1.00e+2F; i += 1) {
    std::cout << std::format("{}", i) << std::endl;
  }
  return 0;
}
