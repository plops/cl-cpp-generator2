#include "xsimd/xsimd.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <memory>
#include <numeric>
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
    auto stat{[&](auto fitres, auto filter) {
      auto data{Vec(fitres.size())};
      std::transform(fitres.begin(), fitres.end(), data.begin(), filter);
      auto mean{std::accumulate(data.begin(), data.end(), 0.F)};
      mean /= data.size();
      auto sq_sum{
          std::inner_product(data.begin(), data.end(), data.begin(), 0.F)};
      auto stdev{std::sqrt((sq_sum / data.size()) - (mean * mean))};
      return std::make_pair(mean, stdev);
    }};
    auto generate_fit{[&]() {
      fill_y();
      return Fitab(x, y);
    }};
    auto fitres{std::vector<Fitab>()};
    fitres.reserve(repeat);
    std::generate_n(std::back_inserter(fitres), repeat, generate_fit);
    auto [am, ad]{stat(fitres, [&](const Fitab &f) { return f.a; })};
    auto [bm, bd]{stat(fitres, [&](const Fitab &f) { return f.b; })};
    return std::make_tuple(am, ad, bm, bd);
  }};
  for (decltype(0 + 30 + 1) i = 0; i < 30; i += 1) {
    auto a{0.30F + 1.00e-2F * dis(gen)};
    auto b{17 + 0.10F * dis(gen)};
    auto sig{3 + 0.10F * dis(gen)};
    auto [am, ad, bm, bd]{lin(18, a, b, sig, 100)};
    std::cout << std::format(
        "( :a '{}' :b '{}' :sig '{}' :am '{}' :ad '{}' :bm '{}' :bd '{}')\n", a,
        b, sig, am, ad, bm, bd);
  }
  return 0;
}
