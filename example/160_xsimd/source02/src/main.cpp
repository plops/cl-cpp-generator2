#include "xsimd/xsimd.hpp"
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <memory>
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
  int ndata;
  Scalar a, b, siga, sigb, chi2, sigdat;
  VecI &x, &y;
};

int main(int argc, char **argv) {
  auto gen{std::mt19937(std::random_device{}())};
  auto dis{std::normal_distribution<float>(0.F, 1.0F)};
  auto lin{[&](auto n, auto a, auto b, auto sig, auto repeat) {
    auto x{Vec(n)};
    auto y{Vec(n)};
    auto fill_x{[&]() { std::iota(x.begin(), x.end(), 0.F); }};
    auto fill_y{[&]() {
      for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        y[i] = dis(gen) + b + a * x[i];
      }
    }};
    fill_x();
    auto ares{Vec(repeat)};
    auto bres{Vec(repeat)};
    auto stat{[&](auto res) {
      auto mean{0.F};
      auto std{0.F};
      for (auto &&r : res) {
        mean += r;
      }
      mean /= repeat;
      for (auto &&r : res) {
        auto d{mean - r};
        std += d * d;
      }
      std /= (repeat - 1.0F);
      std = std::sqrt(std);
      return std::make_shared<Vec>({mean, std});
    }};
    for (decltype(0 + repeat + 1) j = 0; j < repeat; j += 1) {
      fill_y();
      auto f{Fitab(x, y)};
      ares[j] = f.a;
      bres[j] = f.b;
    }
    stat(ares);
    stat(bres);
  }};
  for (decltype(0 + 30 + 1) i = 0; i < 30; i += 1) {
    auto a{0.30F + 1.00e-2F * dis(gen)};
    auto b{17 + 0.10F * dis(gen)};
    auto sig{3 + 0.10F * dis(gen)};
    lin(18, a, b, sig, 100);
  }
  return 0;
}
