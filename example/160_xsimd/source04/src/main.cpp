#include "xsimd/xsimd.hpp"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <valarray>
#include <vector>
using namespace std;
using namespace std::chrono;
using namespace xsimd;
using Scalar = float;
using AVec = valarray<Scalar>;
using AVecI = const AVec;
using Vec = std::vector<Scalar, xsimd::default_allocator<Scalar>>;
using VecI = const Vec;
using Batch = xsimd::batch<Scalar, avx2>;
using Timebase = std::chrono::milliseconds;
constexpr int N = 8 * 1024;

Scalar fun_valarray(AVecI &a, AVecI &b, Scalar c) {
  return std::pow(a * (b + c), 2).sum();
}

Scalar fun_simd(VecI &a, VecI &b, Scalar c) {
  auto inc{Batch::size};
  auto size{a.size()};
  auto sum{Scalar(0.F)};
  auto vec_size{size - (size % inc)};
  // size for which vecotorization is possible
  for (std::size_t i = 0; i < vec_size; i += inc) {
    auto avec{Batch::load_aligned(&a[i])};
    auto bvec{Batch::load_aligned(&b[i])};
    auto rvec{pow(avec * (c + bvec), 2)};
    sum += reduce_add(rvec);
  }
  return sum;
}

int main(int argc, char **argv) {
  {
    auto a{AVec(N)};
    auto b{AVec(N)};
    for (decltype(0 + N + 1) i = 0; i < N; i += 1) {
      a[i] = sin(0.10F * i);
      b[i] = 2.0F;
    }
    const auto start{high_resolution_clock::now()};
    auto res{0.F};
    for (decltype(0 + 100000 + 1) i = 0; i < 100000; i += 1) {
      auto c{1.00e-4F * i};
      res += fun_valarray(a, b, c);
    }
    const auto end{high_resolution_clock::now()};
    const auto duration{duration_cast<Timebase>(end - start)};
    std::cout << std::format(
        "( :fun_valarray(a, b, 0.10F) '{}' :res '{}' :duration '{}')\n",
        fun_valarray(a, b, 0.10F), res, duration);
  }
  {
    auto a{Vec(N)};
    auto b{Vec(N)};
    for (decltype(0 + N + 1) i = 0; i < N; i += 1) {
      a[i] = sin(0.10F * i);
      b[i] = 2.0F;
    }
    const auto start{high_resolution_clock::now()};
    auto res{0.F};
    for (decltype(0 + 100000 + 1) i = 0; i < 100000; i += 1) {
      auto c{1.00e-4F * i};
      res += fun_simd(a, b, c);
    }
    const auto end{high_resolution_clock::now()};
    const auto duration{duration_cast<Timebase>(end - start)};
    std::cout << std::format(
        "( :fun_simd(a, b, 0.10F) '{}' :res '{}' :duration '{}')\n",
        fun_simd(a, b, 0.10F), res, duration);
  }
  return 0;
}
