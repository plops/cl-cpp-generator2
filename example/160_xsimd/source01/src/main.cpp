#include "xsimd/xsimd.hpp"
#include <iostream>
using namespace xsimd;

int main(int argc, char **argv) {
  auto a{batch<double, avx>(1.50F, 2.50F, 3.50F, 4.50F)};
  auto b{batch<double, avx>(2.50F, 3.50F, 4.50F, 5.50F)};
  auto mean{0.50F * (a + b)};
  std::cout << std::format("( :mean '{}')\n", mean);
  return 0;
}
