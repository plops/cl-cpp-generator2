#include <armadillo>
#include <format>
#include <iostream>
using namespace arma;

int main(int argc, char **argv) {
  std::cout << std::format("start\n");
  arma_rng::set_seed_random();
  auto A{randn(5, 5)};
  auto B{mat(pinv(A))};
  auto C{mat(inv(A))};
  for (const auto &b : B) {
    std::cout << std::format(" b='{}'\n", b);
  }
  for (const auto &c : C) {
    std::cout << std::format(" c='{}'\n", c);
  }
  return 0;
}
