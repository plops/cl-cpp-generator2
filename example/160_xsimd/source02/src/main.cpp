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
  auto gen{std::mt19937(42)};
  auto gauss{[&](Scalar mu, Scalar sig) -> Scalar {
    // Leva Gaussian Noise with ratio of uniforms
    Scalar u, v, x, y, q;
    do {
      u = gen();
      v = 1.71560F * (gen() - 0.50F);
      x = (u - 0.4498710F);
      y = std::abs(v) + 0.3865950F;
      q = x * x + y * ((0.1960F * y) - (0.254720F * x));
    } while (0.275970F < q &&
             (0.278460F < q || -4.0F * std::log(u) * u * u < v * v));
    return mu + sig * (v / u);
  }};
  return 0;
}
