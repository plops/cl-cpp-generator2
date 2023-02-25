#include <array>
#include <chrono>
#include <fmt/core.h>
#include <immintrin.h>
#include <iostream>
#define ARRAY_SIZE 1000000

int main(int argc, char **argv) {
  fmt::print("14:43:52 of Saturday, 2023-02-25 (GMT+1)\n");
  std::srand(std::time(nullptr));
  auto array = std::array<int, ARRAY_SIZE>();
  for (auto &e : array) {
    e = std::rand();
  };

  // if rdpmc crashes, run this: echo 2 | sudo tee /sys/devices/cpu/rdpmc

  auto cycles = __rdpmc((1 << 30 + 0));

  return 0;
}
