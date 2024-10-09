#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using Scalar = float;
using Vec = std::vector<Scalar>;
using VecI = const Vec;
// From Numerical Recipes
class Fitab {
public:
  Fitab(VecI &xx, VecI &yy) : ndata{static_cast<int>(xx.size())}, x{xx}, y{yy} {
    const auto sx{std::accumulate(x.begin(), x.end(), 0.F)};
    const auto sy{std::accumulate(y.begin(), y.end(), 0.F)};
    const auto ss{static_cast<Scalar>(ndata)};
    const auto sxoss{sx / ss};
    const auto st2{
        std::accumulate(x.begin(), x.end(), 0.F, [&](auto accum, auto xi) {
          return accum + std::pow(xi - sxoss, 2.0F);
        })};
    for (decltype(0 + ndata + 1) i = 0; i < ndata; i += 1) {
      const auto tt{(x[i]) - sxoss};
      b += tt * y[i];
    }
    // solve for a, b, sigma_a and sigma_b
    b /= st2;
    a = ((sy - (b * sx)) / ss);
    siga = std::sqrt((1.0F + ((sx * sx) / (ss * st2))) / ss);
    sigb = std::sqrt(1.0F / st2);
    // compute chi2
    chi2 = std::inner_product(
        x.begin(), x.end(), y.begin(), 0.F,
        [&](auto accum, auto value) { return accum + value; },
        [this](auto xi, auto yi) {
          auto p{yi - a - (b * xi)};
          return p * p;
        });
    if (2 < ndata) {
      sigdat = std::sqrt(chi2 / (static_cast<Scalar>(ndata) - 2.0F));
    }
    siga *= sigdat;
    sigb *= sigdat;
  }
  int ndata{0};
  Scalar a{.0f}, b{.0f}, siga{.0f}, sigb{.0f}, chi2{.0f}, sigdat{.0f};
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

std::string printStat(std::tuple<Scalar, Scalar, Scalar, Scalar> m_md_d_dd) {
  auto [m, md, d, dd]{m_md_d_dd};
  const auto mprecision{getSignificantDigits(md)};
  const auto dprecision{getSignificantDigits(dd)};
  const auto fmtm{std::string("{:.") + std::to_string(mprecision) + "f}"};
  const auto fmtd{std::string("{:.") + std::to_string(dprecision) + "f}"};
  const auto format_str{fmtm + "±" + fmtd};
  return std::vformat(format_str, std::make_format_args(m, d));
}

int main(int argc, char **argv) {
  auto gen{std::mt19937(std::random_device{}())};
  auto dis{std::normal_distribution<float>(0.F, 1.0F)};
  auto lin{[&](auto n, auto A, auto B, auto Sig, auto repeat) {
    auto x{Vec(n)};
    auto y{Vec(n)};
    auto fill_x{[&]() { std::iota(x.begin(), x.end(), 1.0F); }};
    fill_x();
    auto stat{[&](auto fitres, auto filter) {
      // Numerical Recipes 14.1.2
      auto data{Vec(fitres.size())};
      data.resize(fitres.size());
      std::transform(fitres.begin(), fitres.end(), data.begin(), filter);
      const auto N{static_cast<Scalar>(data.size())};
      const auto mean{std::accumulate(data.begin(), data.end(), 0.F) / N};
      const auto stdev{
          std::sqrt((std::accumulate(data.begin(), data.end(), 0.F,
                                     [mean](auto acc, auto xi) {
                                       return acc + std::pow(xi - mean, 2.0F);
                                     }) -
                     (std::pow(std::accumulate(data.begin(), data.end(), 0.F,
                                               [mean](auto acc, auto xi) {
                                                 return acc + (xi - mean);
                                               }),
                               2.0F) /
                      N)) /
                    (N - 1.0F))};
      const auto mean_stdev{stdev / std::sqrt(N)};
      const auto stdev_stdev{stdev / std::sqrt(2 * N)};
      return std::make_tuple(mean, mean_stdev, stdev, stdev_stdev);
    }};
    auto generate_fit{[&]() {
      std::transform(x.begin(), x.end(), y.begin(),
                     [&](Scalar xi) { return Sig * dis(gen) + A + B * xi; });
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
  for (decltype(0 + 3 + 1) i = 0; i < 3; i += 1) {
    auto A{17 + 0.10F * dis(gen)};
    auto B{0.30F + 1.00e-2F * dis(gen)};
    auto Sig{10.F};
    auto [a, b, siga, sigb, chi2, sigdat]{lin(133, A, B, Sig, 17)};
    const auto pa{printStat(a)};
    const auto pb{printStat(b)};
    const auto psiga{printStat(siga)};
    const auto psigb{printStat(sigb)};
    const auto pchi2{printStat(chi2)};
    const auto psigdat{printStat(sigdat)};
    std::cout << std::format(
        "( :A '{}' :B '{}' :Sig '{}' :pa '{}' :pb '{}' :psiga '{}' :psigb '{}' "
        ":pchi2 '{}' :psigdat '{}')\n",
        A, B, Sig, pa, pb, psiga, psigb, pchi2, psigdat);
  }
  return 0;
}
