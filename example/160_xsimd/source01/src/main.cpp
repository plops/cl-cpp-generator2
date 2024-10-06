#include "xsimd/xsimd.hpp"
#include <cstddef>
#include <format>
#include <iostream>
#include <vector>
using namespace xsimd;
using Voc = std::vector<float, xsimd::default_allocator<float>>;
using Batch = xsimd::batch<float, avx2>;

void mean(const Voc &a, const Voc &b, Voc &res) {
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

void transpose(const Voc &a, Voc &b) {
  auto N{std::sqrt(a.size())};
  for (decltype(0 + N + 1) i = 0; i < N; i += 1) {
    for (decltype(0 + N + 1) j = 0; j < N; j += 1) {
      auto block{Batch::load_aligned(&a[(i * N + j)])};
      block.store_aligned(&b[(j * N + i)]);
    }
  }
}

int main(int argc, char **argv) {
  auto a{Voc({1.50F, 2.50F, 3.50F, 4.50F, 1, 2, 3, 4})};
  auto b{Voc({2.50F, 3.50F, 4.50F, 5.50F, 2, 3, 4, 5})};
  auto e{Voc({2.0F, 3.0F, 4.0F, 5.0F, 1.50F, 2.50F, 3.50F, 4.50F})};
  auto r{Voc(b.size())};
  mean(a, b, r);
  std::cout << std::format(
      "( :r[0] '{}' :r[1] '{}' :r[2] '{}' :r[3] '{}' :r[4] '{}' :r[5] '{}' "
      ":r[6] '{}' :r[7] '{}' :(r[0])-(e[0]) '{}' :(r[1])-(e[1]) '{}' "
      ":(r[2])-(e[2]) '{}' :(r[3])-(e[3]) '{}' :(r[4])-(e[4]) '{}' "
      ":(r[5])-(e[5]) '{}' :(r[6])-(e[6]) '{}' :(r[7])-(e[7]) '{}')\n",
      r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], (r[0]) - (e[0]),
      (r[1]) - (e[1]), (r[2]) - (e[2]), (r[3]) - (e[3]), (r[4]) - (e[4]),
      (r[5]) - (e[5]), (r[6]) - (e[6]), (r[7]) - (e[7]));
  {
    auto a{
        Voc({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
             48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63})};
    auto e{Voc({0, 8,  16, 24, 32, 40, 48, 56, 1, 9,  17, 25, 33, 41, 49, 57,
                2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
                4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
                6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63})};
    auto r{Voc(b.size())};
    transpose(a, r);
    std::cout << std::format(
        "( :r[0] '{}' :r[1] '{}' :r[2] '{}' :r[3] '{}' :r[4] '{}' :r[5] '{}' "
        ":r[6] '{}' :r[7] '{}' :(r[0])-(e[0]) '{}' :(r[1])-(e[1]) '{}' "
        ":(r[2])-(e[2]) '{}' :(r[3])-(e[3]) '{}' :(r[4])-(e[4]) '{}' "
        ":(r[5])-(e[5]) '{}' :(r[6])-(e[6]) '{}' :(r[7])-(e[7]) '{}')\n",
        r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], (r[0]) - (e[0]),
        (r[1]) - (e[1]), (r[2]) - (e[2]), (r[3]) - (e[3]), (r[4]) - (e[4]),
        (r[5]) - (e[5]), (r[6]) - (e[6]), (r[7]) - (e[7]));
  }
  return 0;
}
