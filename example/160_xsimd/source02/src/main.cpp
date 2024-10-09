#include <algorithm>
#include <cmath>
#include <execution>
#include <format>
#include <iostream>
#include <numeric>
#include <popl.hpp>
#include <random>
#include <thread>
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
#pragma omp parallel for reduction(+ : b)
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
  Scalar a{.0f}, siga{.0f}, b{.0f}, sigb{.0f}, chi2{.0f}, sigdat{.0f};
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
  const auto rel{1.00e+2F * (d / m)};
  const auto mprecision{getSignificantDigits(md)};
  const auto dprecision{getSignificantDigits(dd)};
  const auto rprecision{getSignificantDigits(rel)};
  const auto fmtm{std::string("{:.") + std::to_string(mprecision) + "f}"};
  const auto fmtd{std::string("{:.") + std::to_string(dprecision) + "f}"};
  const auto fmtr{std::string(" ({:.") + std::to_string(rprecision) + "f}%)"};
  const auto format_str{fmtm + "±" + fmtd + fmtr};
  return std::vformat(format_str, std::make_format_args(m, d, rel));
}

Scalar select(const int k, Vec &arr) {
  // This implementation uses the STL and will not fall under the strict license
  // of Numerical Recipes
  if (0 <= k && k < arr.size()) {
    std::nth_element(arr.begin(), arr.begin() + k, arr.end());
    return arr[k];
  }
  throw std::out_of_range("Invalid index for selection");
}

int main(int argc, char **argv) {
  auto op{popl::OptionParser("allowed options")};
  auto numberRepeats{int(64)};
  auto numberPoints{int(1024)};
  auto numberTrials{int(3)};
  auto helpOption{op.add<popl::Switch>("h", "help", "produce help message")};
  auto verboseOption{
      op.add<popl::Switch>("v", "verbose", "produce verbose output")};
  auto numberRepeatsOption{op.add<popl::Value<int>>(
      "r", "numberRepeats", "parameter", 64, &numberRepeats)};
  auto numberPointsOption{op.add<popl::Value<int>>(
      "p", "numberPoints", "parameter", 1024, &numberPoints)};
  auto numberTrialsOption{op.add<popl::Value<int>>(
      "d", "numberTrials", "parameter", 3, &numberTrials)};
  op.parse(argc, argv);
  if (helpOption->count()) {
    std::cout << op << std::endl;
    exit(0);
  }
  std::cout << std::format("(:std::thread::hardware_concurrency() '{}')\n",
                           std::thread::hardware_concurrency());
  auto gen{std::mt19937(std::random_device{}())};
  auto dis{std::normal_distribution<float>(0.F, 1.0F)};
  auto lin{[&](auto n, auto A, auto B, auto Sig, auto repeat) {
    auto x{Vec(n)};
    auto y{Vec(n)};
    auto fill_x{[&]() { std::iota(x.begin(), x.end(), 1.0F); }};
    fill_x();
    auto stat_median{[&](auto fitres, auto filter) {
      // compute median and median absolute deviation Numerical recipes 8.5
      // and 14.1.4
      auto data{Vec(fitres.size())};
      data.resize(fitres.size());
      std::transform(fitres.begin(), fitres.end(), data.begin(), filter);
      const auto N{static_cast<Scalar>(data.size())};
      const auto median{select((static_cast<int>(data.size()) - 1) / 2, data)};
      const auto adev{std::accumulate(data.begin(), data.end(), 0.F,
                                      [median](auto acc, auto xi) {
                                        return acc + std::abs(xi - median);
                                      }) /
                      N};
      const auto mean_stdev{adev / std::sqrt(N)};
      const auto stdev_stdev{adev / std::sqrt(2 * N)};
      return std::make_tuple(median, mean_stdev, adev, stdev_stdev);
    }};
    auto generate_fit{[&]() {
      std::transform(x.begin(), x.end(), y.begin(),
                     [&](Scalar xi) { return Sig * dis(gen) + A + B * xi; });
      return Fitab(x, y);
    }};
    auto fitres{std::vector<Fitab>()};
    fitres.reserve(repeat);
    std::generate_n(std::back_inserter(fitres), repeat, generate_fit);
    auto a{stat_median(fitres, [&](const Fitab &f) { return f.a; })};
    auto siga{stat_median(fitres, [&](const Fitab &f) { return f.siga; })};
    auto b{stat_median(fitres, [&](const Fitab &f) { return f.b; })};
    auto sigb{stat_median(fitres, [&](const Fitab &f) { return f.sigb; })};
    auto chi2{stat_median(fitres, [&](const Fitab &f) { return f.chi2; })};
    auto sigdat{stat_median(fitres, [&](const Fitab &f) { return f.sigdat; })};
    return std::make_tuple(a, siga, b, sigb, chi2, sigdat);
  }};
  for (decltype(0 + 3 + 1) i = 0; i < 3; i += 1) {
    const auto dA{0.10F};
    const auto A{17.F + dA * dis(gen)};
    const auto dB{1.00e-2F};
    const auto B{0.30F + dB * dis(gen)};
    const auto Sig{10.F};
    auto [a, siga, b, sigb, chi2, sigdat]{lin(8133, A, B, Sig, 8917)};
    const auto pa{printStat(a)};
    const auto psiga{printStat(siga)};
    const auto pb{printStat(b)};
    const auto psigb{printStat(sigb)};
    const auto pchi2{printStat(chi2)};
    const auto psigdat{printStat(sigdat)};
    std::cout << std::format(
        "(:A '{}' :dA '{}' :B '{}' :dB '{}' :Sig '{}' :pa '{}' :psiga '{}' :pb "
        "'{}' :psigb '{}' :pchi2 '{}' :psigdat '{}')\n",
        A, dA, B, dB, Sig, pa, psiga, pb, psigb, pchi2, psigdat);
  }
  return 0;
}
