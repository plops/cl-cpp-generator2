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

int getSignificantDigits(Scalar num) {
  if (num == 0.F) {
    return 1;
  }
  if (num < 0) {
    num = -num;
  }
  auto significantDigits{0};
  while (num <= 1.0F) {
    num *= 10.F;
    significantDigits++;
  }
  return significantDigits;
}

std::string printStat(std::pair<Scalar, Scalar> md) {
  auto [m, d]{md};
  auto precision{getSignificantDigits(d)};
  auto fmtm{std::string("{:.") + std::to_string(1 + precision) + "f}"};
  auto fmtd{std::string("{:.") + std::to_string(precision) + "f}"};
  const std::string format_str{fmtm + " ± " + fmtd};
  return std::vformat(format_str, std::make_format_args(m, d));
}

int main(int argc, char **argv) {
  auto gen{std::mt19937(std::random_device{}())};
  auto dis{std::normal_distribution<float>(0.F, 1.0F)};
  auto lin{[&](auto n, auto A, auto B, auto Sig, auto repeat) {
    auto x{Vec(n)};
    auto y{Vec(n)};
    auto fill_x{[&]() { std::iota(x.begin(), x.end(), 0.F); }};
    auto fill_y{[&]() {
      for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        y[i] = Sig * dis(gen) + A + B * x[i];
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
    auto a{stat(fitres, [&](const Fitab &f) { return f.a; })};
    auto b{stat(fitres, [&](const Fitab &f) { return f.b; })};
    auto siga{stat(fitres, [&](const Fitab &f) { return f.siga; })};
    auto sigb{stat(fitres, [&](const Fitab &f) { return f.sigb; })};
    auto chi2{stat(fitres, [&](const Fitab &f) { return f.chi2; })};
    auto sigdat{stat(fitres, [&](const Fitab &f) { return f.sigdat; })};
    return std::make_tuple(a, b, siga, sigb, chi2, sigdat);
  }};
  for (decltype(0 + 30 + 1) i = 0; i < 30; i += 1) {
    auto A{17 + 0.10F * dis(gen)};
    auto B{0.30F + 1.00e-2F * dis(gen)};
    auto Sig{3 + 0.10F * dis(gen)};
    auto [a, b, siga, sigb, chi2, sigdat]{lin(18, A, B, Sig, 100)};
    std::cout << std::format(
        "( :A '{}' :B '{}' :Sig '{}' :printStat(a) '{}' :printStat(b) '{}' "
        ":printStat(siga) '{}' :printStat(sigb) '{}' :printStat(chi2) '{}' "
        ":printStat(sigdat) '{}')\n",
        A, B, Sig, printStat(a), printStat(b), printStat(siga), printStat(sigb),
        printStat(chi2), printStat(sigdat));
  }
  return 0;
}
