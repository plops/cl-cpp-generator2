#include <array>
#include <chrono>
#include <iostream>
constexpr int N = 1048576;

constexpr int ITER = 100000;

int main(int argc, char **argv) {
  auto array = std::array<int, N>();
  for (auto i = 0; i < N; i += 1) {
    array[i] = i;
  }
  auto sum = int(0);
  auto start = std::chrono::high_resolution_clock::now();
  for (auto j = 0; j < ITER; j += 1) {
    for (auto i = 0; i < N; i += 1) {
      sum += array[i];
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = ((end) - (start));
  std::cout << "elapsed: " << elapsed.count() << " ns" << std::endl;

  return 0;
}
