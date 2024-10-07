#include "xsimd/xsimd.hpp"
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <random>
#include <vector>
using namespace xsimd;
using Scalar = float;
using ScalarI = const Scalar;
using XVec = std::vector<Scalar, xsimd::default_allocator<Scalar>>;
using XBatch = xsimd::batch<Scalar, avx2>;
using Vec = std::vector<Scalar>;
using VecI = const Vec;
class Fitab {
public:
  Fitab(VecI &xx, VecI &yy)
      : ndata{static_cast<int>(xx.size())}, x{xx}, y{yy}, b{.0f}, chi2{.0f},
        sigdat{.0f} {
    auto sx{.0f};
    auto sy{.0f};
    for (decltype(0 + ndata + 1) i = 0; i < ndata; i += 1) {
      sx += x[i];
      sy += y[i];
    }
    auto ss{static_cast<Scalar>(ndata)};
    auto sxoss{sx / ss};
    auto st2{.0f};
    auto tt{.0f};
    for (decltype(0 + ndata + 1) i = 0; i < ndata; i += 1) {
      tt += ((x[i]) - sxoss);
      st2 += tt * tt;
      b += tt * y[i];
    }
    // solve for a, b, sigma_a and sigma_b
    b /= st2;
    a = ((sy - (b * sx)) / ss);
    siga = std::sqrt((1.0F + ((sx * sx) / (ss * st2))) / ss);
    sigb = std::sqrt(1.0F / st2);
    // compute chi2
    for (decltype(0 + ndata + 1) i = 0; i < ndata; i += 1) {
      auto p{(y[i]) - a - (b * x[i])};
      chi2 += p * p;
    }
    if (2 < ndata) {
      sigdat = std::sqrt(chi2 / (static_cast<Scalar>(ndata) - 2.0F));
    }
    siga *= sigdat;
    sigb *= sigdat;
  }

private:
  int ndata;
  Scalar a, b, siga, sigb, chi2, sigdat;
  VecI &x, &y;
};

int main(int argc, char **argv) {
  auto gen{std::mt19937(std::random_device{}())};
  auto dis{std::normal_distribution<float>(0.F, 1.0F)};
  auto lin{[&](auto n, auto a, auto b, auto sig) {
    auto x{Vec(n)};
    auto y{Vec(n)};
    std::iota(x.begin(), x.end(), 0.F);
    for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
      y[i] = dis(gen) + b + a * x[i];
    }
    auto f{Fitab(x, y)};
  }};
  for (decltype(0 + 30 + 1) i = 0; i < 30; i += 1) {
    auto a{0.30F + 1.00e-2F * dis(gen)};
    auto b{17 + 0.10F * dis(gen)};
    auto sig{3 + 0.10F * dis(gen)};
    lin(18, a, b, sig);
  }
  return 0;
}
