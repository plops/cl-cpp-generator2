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
constexpr int N = 8 * 1024;

Scalar fun_valarray(AVecI &a, AVecI &b) { return std::pow(a * b, 2).sum(); }

Scalar fun_simd(VecI &a, VecI &b) {
  auto inc{Batch::size};
  auto size{a.size()};
  auto sum{Scalar(0.F)};
  auto vec_size{size - (size % inc)};
  // size for which vecotorization is possible
  for (std::size_t i = 0; i < vec_size; i += inc) {
    auto avec{Batch::load_aligned(&a[i])};
    auto bvec{Batch::load_aligned(&b[i])};
    auto rvec{pow(avec * bvec, 2)};
    sum += reduce_add(rvec);
  }
  return sum;
}

int main(int argc, char **argv) {
  auto aa{AVec(N)};
  auto ab{AVec(N)};
  for (decltype(0 + N + 1) i = 0; i < N; i += 1) {
    aa[i] = sin(0.10F * i);
    ab[i] = 2.0F;
  }
  std::cout << std::format("( :fun_valarray(aa, ab) '{}')\n",
                           fun_valarray(aa, ab));
  auto a{Vec(N)};
  auto b{Vec(N)};
  for (decltype(0 + N + 1) i = 0; i < N; i += 1) {
    a[i] = sin(0.10F * i);
    b[i] = 2.0F;
  }
  std::cout << std::format("( :fun_simd(a, b) '{}')\n", fun_simd(a, b));
  return 0;
}
