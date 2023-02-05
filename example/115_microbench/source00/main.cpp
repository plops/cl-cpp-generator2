#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
constexpr int64_t N = 1048576;

constexpr int64_t ITER = 10000;

int main(int argc, char **argv) {
  auto array = std::array<int, N>();
  for (auto i = 0; i < N; i += 1) {
    array[i] = i;
  }
  auto sum = int32_t(0);
  auto start = std::chrono::high_resolution_clock::now();
  for (auto j = 0; j < ITER; j += 1) {
    for (auto i = 0; i < N; i += 1) {
      sum += array[i];
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = ((end) - (start));
  std::cout << "time per iteration and element: "
            << ((elapsed.count()) / (static_cast<double>((N * ITER)))) << " ns"
            << std::endl;
  std::cout << "sum: " << sum << " N: " << N << std::endl;

  return 0;
}
