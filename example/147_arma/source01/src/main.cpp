#include <armadillo>
#include <format>
#include <iostream>
using namespace arma;

int main(int argc, char **argv) {
  std::cout << std::format("start\n");
  arma_rng::set_seed_random();
  auto A{randn(5, 5)};
  for (const auto &a : A) {
    std::cout << std::format(" a='{}'\n", a);
  }
  return 0;
}
