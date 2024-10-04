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
  auto charp{static_cast<char *>(mb.allocate<char>())};
  auto doublep{static_cast<double *>(mb.allocate<double>())};
  auto shortp{static_cast<short *>(mb.allocate<short>())};
  auto intp{static_cast<int *>(mb.allocate<int>())};
  auto boolp{static_cast<bool *>(mb.allocate<bool>())};
  std::cout << "char:" << (reinterpret_cast<char *>(charp) - &mb.d_buffer[0])
            << std::endl;
  std::cout << "double:"
            << (reinterpret_cast<char *>(doublep) - &mb.d_buffer[0])
            << std::endl;
  std::cout << "short:" << (reinterpret_cast<char *>(shortp) - &mb.d_buffer[0])
            << std::endl;
  std::cout << "int:" << (reinterpret_cast<char *>(intp) - &mb.d_buffer[0])
            << std::endl;
  std::cout << "bool:" << (reinterpret_cast<char *>(boolp) - &mb.d_buffer[0])
            << std::endl;
  for (decltype(3.0F) i = 0; i < 3.0F; i += 1) {
    std::cout << std::format("{}", i) << std::endl;
  }
  return 0;
}
