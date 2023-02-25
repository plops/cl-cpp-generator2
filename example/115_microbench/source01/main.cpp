#include <array>
#include <chrono>
#include <iostream>
#define ARRAY_SIZE 1000000

int main(int argc, char **argv) {
  std::srand(std::time(nullptr));
  auto array = std::array<int, ARRAY_SIZE>();
  for (auto &e : array) {
    e = std::rand();
  };

  return 0;
}
