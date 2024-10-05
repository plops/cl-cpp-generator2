#include "xsimd/xsimd.hpp"
#include <cstddef>
#include <format>
#include <iostream>
#include <vector>
using namespace xsimd;
using Voc = std::vector<float, xsimd::default_allocator<float>>;

void mean(const Voc &a, const Voc &b, Voc &res) {
  using Batch = xsimd::batch<float, avx2>;
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
  auto a{Voc({1.50F, 2.50F, 3.50F, 4.50F, 1, 2, 3, 4})};
  auto b{Voc({2.50F, 3.50F, 4.50F, 5.50F, 2, 3, 4, 5})};
  auto r{Voc(b.size())};
  mean(a, b, r);
  std::cout << std::format("( :r[0] '{}' :r[1] '{}' :r[2] '{}' :r[3] '{}' "
                           ":r[4] '{}' :r[5] '{}' :r[6] '{}' :r[7] '{}')\n",
                           r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
  return 0;
}
