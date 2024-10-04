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
  template <typename T> void *allocate() { return 0; };
};

int main(int argc, char **argv) {
  for (decltype(1.00e+2F) i = 0; i < 1.00e+2F; i += 1) {
    std::cout << std::format("{}", i) << std::endl;
  }
  return 0;
}
