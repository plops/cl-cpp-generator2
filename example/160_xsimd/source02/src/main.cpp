#include "xsimd/xsimd.hpp"
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
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
      : ndata{xx.size()}, x{xx}, y{yy}, chi2{.0f}, sigdat{.0f} {
    auto sx{.0f};
    auto sy{.0f};
    for (decltype(0 + ndata + 1) i = 0; i < ndata; i += 1) {
      sx += x[i];
      sy += y[i];
    }
    auto ss{static_cast<Scalar>(ndata)};
    auto sxoss{sx / ss};
    auto ss{.0f};
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
      sigdat = std::sqrt(chi2 / (ndata - 2));
    }
    siga *= sigdat;
    sigb *= sigdat;
  }

private:
  int ndata;
  Scalar a, b, siga, sigb, chi2, sigdat;
  VecI &x, &y;
};

int main(int argc, char **argv) { return 0; }
