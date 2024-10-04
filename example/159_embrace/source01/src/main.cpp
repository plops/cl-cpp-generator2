// Some experiments while reading Lakos: Embracing Modern C++ Safely (2021)
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <vector>
std::atomic<int64_t> g_allocCount{0};

void *operator new(std::size_t size) {
  g_allocCount.fetch_add(size, std::memory_order_relaxed);
  return malloc(size);
}

void operator delete(void *p, std::size_t size) {
  g_allocCount.fetch_sub(size, std::memory_order_relaxed);
  free(p);
}

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
// emulate named parameters
auto computeGen{[]() {
  class ComputeArgs {
  public:
    float sigma{1.2};
    int maxIterations{100};
    float tol{.001};
  };
  return [&](const ComputeArgs &args) -> float {
    std::cout << std::format(
        "( :args.sigma '{}' :args.maxIterations '{}' :args.tol '{}')\n",
        args.sigma, args.maxIterations, args.tol);
    return args.sigma * args.tol * args.maxIterations;
  };
}};
auto compute{computeGen()};

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
  auto vec{std::vector<int>(12)};
  auto count{100};
  for (auto &&v : vec) {
    v = count++;
  }
  for (decltype(3.0F) i = 0; i < 3.0F; i += 1) {
    std::cout << std::format("{}", i) << std::endl;
  }
  for (decltype(3) i = 0; i < 3; i += 1) {
    std::cout << std::format("{}", vec[i]) << std::endl;
  }
  std::cout << std::format("( :g_allocCount.load() '{}')\n",
                           g_allocCount.load());
  constexpr int nVb = 1'000'000;
  auto vb{std::vector<bool>(nVb)};
  auto sizeofVb{sizeof(vb)};
  auto sizeVb{vb.size()};
  auto bytesVb{0};
  std::cout << std::format(
      "(vector<bool> :sizeVb '{}' :sizeofVb '{}' :bytesVb '{}')\n", sizeVb,
      sizeofVb, bytesVb);
  std::cout << std::format("( :g_allocCount.load() '{}')\n",
                           g_allocCount.load());
  constexpr int nAb = 1'000'000;
  auto ab{std::array<bool, nAb>()};
  auto sizeofAb{sizeof(ab)};
  auto sizeAb{ab.size()};
  auto bytesAb{&ab.data()[(nAb - 1)] - &ab.data()[0]};
  std::cout << std::format(
      "(array<bool> :sizeAb '{}' :sizeofAb '{}' :bytesAb '{}')\n", sizeAb,
      sizeofAb, bytesAb);
  std::cout << std::format("( :g_allocCount.load() '{}')\n",
                           g_allocCount.load());
  // i just read section about initializer-lists
  compute({});
  compute({.maxIterations = 10});
  compute({.maxIterations = 20});
  return 0;
}
