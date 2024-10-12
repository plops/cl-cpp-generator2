#include "xsimd/xsimd.hpp"
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <valarray>
#include <vector>
using namespace std;
using namespace xsimd;
using Scalar = float;
using AVec = valarray<Scalar>;
using AVecI = const AVec;
using Vec = std::vector<Scalar, xsimd::default_allocator<Scalar>>;
using VecI = const Vec;
using Batch = xsimd::batch<Scalar, avx2>;
constexpr int N = 8192;

Scalar fun_valarray(AVecI &a, AVecI &b) { return std::pow(a * b, 2).sum(); }

int main(int argc, char **argv) {
  const auto aa{AVec(N)};
  const auto ab{AVec(N)};
  std::cout << std::format("( :fun_valarray(aa, ab) '{}')\n",
                           fun_valarray(aa, ab));
  return 0;
}
