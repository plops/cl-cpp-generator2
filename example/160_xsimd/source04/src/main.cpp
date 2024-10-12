#include "xsimd/xsimd.hpp"
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

Scalar fun_valarray(AVecI &a, AVecI &b) { return pow(a * b, 2).sum(); }

void fun_simd(VecI &a, VecI &b, Vec &res) {
  auto inc{Batch::size};
  auto size{res.size()};
  auto vec_size{size - (size % inc)};
  // size for which vecotorization is possible
  for (std::size_t i = 0; i < vec_size; i += inc) {
    auto avec{Batch::load_aligned(&a[i])};
    auto bvec{Batch::load_aligned(&b[i])};
    auto rvec{(avec + bvec) * 0.50F};
    rvec.store_aligned(&res[i]);
  }
  // remaining part that can't be vectorized
  for (std::size_t i = vec_size; i < size; i++) {
    res[i] = (a[i] + b[i]) * 0.50F;
  }
}

int main(int argc, char **argv) {
  auto aa{AVec(N)};
  auto ab{AVec(N)};
  std::cout << std::format("( :fun_valarray(aa, ab) '{}')\n",
                           fun_valarray(aa, ab));
  auto a{Vec(N)};
  auto b{Vec(N)};
  std::cout << std::format("( :fun_simd(a, b) '{}')\n", fun_simd(a, b));
  return 0;
}
