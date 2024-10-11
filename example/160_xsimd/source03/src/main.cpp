#define EIGEN_VECTORIZE_AVX2
#define EIGEN_VECTORIZE_AVX
#define EIGEN_VECTORIZE
#define EIGEN_VECTORIZE_FMA
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
#include <popl.hpp>
#include <random>
#include <thread>
#include <vector>
using namespace std;
using namespace Eigen;
using Scalar = float;
using Vec = Matrix<Scalar, 1, Dynamic, RowMajor, 1, 8192>;
using VecI = const Vec;
// From Numerical Recipes
class Fitab {
public:
  Fitab(VecI &xx, VecI &yy) : ndata{static_cast<int>(xx.size())}, x{xx}, y{yy} {
    const auto sx{x.sum()};
    const auto sy{y.sum()};
    const auto ss{static_cast<Scalar>(ndata)};
    const auto sxoss{sx / ss};
    const auto tt{x.array() - sxoss};
    const auto st2{pow(tt, 2).sum()};
    b = ((tt.array() * y.array()) / st2).sum();
    a = ((sy - (b * sx)) / ss);
    siga = sqrt((1.0F + ((sx * sx) / (ss * st2))) / ss);
    sigb = sqrt(1.0F / st2);
    // compute chi2
    chi2 = pow(y.array() - a - (b * x.array()), 2).sum();
    if (2 < ndata) {
      sigdat = sqrt(chi2 / (static_cast<Scalar>(ndata) - 2.0F));
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

string printStat(tuple<Scalar, Scalar, Scalar, Scalar> m_md_d_dd) {
  auto [m, md, d, dd]{m_md_d_dd};
  const auto rel{1.00e+2F * (d / m)};
  const auto mprecision{getSignificantDigits(md)};
  const auto dprecision{getSignificantDigits(dd)};
  const auto rprecision{getSignificantDigits(rel)};
  const auto fmtm{std::string("{:.") + to_string(mprecision) + "f}"};
  const auto fmtd{std::string("{:.") + to_string(dprecision) + "f}"};
  const auto fmtr{std::string(" ({:.") + to_string(rprecision) + "f}%)"};
  const auto format_str{fmtm + "±" + fmtd + fmtr};
  return vformat(format_str, make_format_args(m, d, rel));
}

Scalar select(const int k, Vec &arr) {
  // This implementation uses the STL and will not fall under the strict license
  // of Numerical Recipes
  if (0 <= k && k < arr.size()) {
    nth_element(&arr[0], &arr[0] + k, &arr[0] + arr.size());
    return arr[k];
  }
  throw out_of_range("Invalid index for selection");
}

int main(int argc, char **argv) {
  auto op{popl::OptionParser("allowed options")};
  auto numberRepeats{int(64)};
  auto numberPoints{int(1024)};
  auto numberTrials{int(3)};
  auto generatorSlope{Scalar(0.30F)};
  auto generatorDeltaSlope{Scalar(1.00e-2F)};
  auto generatorIntercept{Scalar(17.F)};
  auto generatorDeltaIntercept{Scalar(0.10F)};
  auto generatorSigma{Scalar(10.F)};
  auto helpOption{op.add<popl::Switch>("h", "help", "produce help message")};
  auto verboseOption{
      op.add<popl::Switch>("v", "verbose", "produce verbose output")};
  auto meanOption{op.add<popl::Switch>(
      "m", "mean",
      "Print mean and standard deviation statistics, otherwise print median "
      "and mean absolute deviation from it")};
  auto numberRepeatsOption{op.add<popl::Value<int>>(
      "r", "numberRepeats", "parameter", 64, &numberRepeats)};
  auto numberPointsOption{op.add<popl::Value<int>>(
      "p", "numberPoints", "parameter", 1024, &numberPoints)};
  auto numberTrialsOption{op.add<popl::Value<int>>(
      "d", "numberTrials", "parameter", 3, &numberTrials)};
  auto generatorSlopeOption{op.add<popl::Value<Scalar>>(
      "B", "generatorSlope", "parameter", 0.30F, &generatorSlope)};
  auto generatorDeltaSlopeOption{op.add<popl::Value<Scalar>>(
      "b", "generatorDeltaSlope", "parameter", 1.00e-2F, &generatorDeltaSlope)};
  auto generatorInterceptOption{op.add<popl::Value<Scalar>>(
      "A", "generatorIntercept", "parameter", 17.F, &generatorIntercept)};
  auto generatorDeltaInterceptOption{
      op.add<popl::Value<Scalar>>("a", "generatorDeltaIntercept", "parameter",
                                  0.10F, &generatorDeltaIntercept)};
  auto generatorSigmaOption{op.add<popl::Value<Scalar>>(
      "s", "generatorSigma", "parameter", 10.F, &generatorSigma)};
  op.parse(argc, argv);
  if (helpOption->is_set()) {
    cout << op << endl;
    exit(0);
  }
  std::cout << std::format("(:thread::hardware_concurrency() '{}')\n",
                           thread::hardware_concurrency());
  auto gen{mt19937(random_device{}())};
  auto dis{normal_distribution<float>(0.F, 1.0F)};
  auto lin{[&](auto n, auto A, auto B, auto Sig, auto repeat) {
    auto x{Vec(n)};
    auto y{Vec(n)};
    auto fill_x{[&]() { iota(&x[0], &x[0] + x.size(), 1.0F); }};
    fill_x();
    auto stat_median{[&](const auto &fitres,
                         auto filter) -> tuple<Scalar, Scalar, Scalar, Scalar> {
      // compute median and median absolute deviation Numerical recipes 8.5
      // and 14.1.4
      auto data{Vec(fitres.size())};
      data.resize(fitres.size());
      transform(fitres.begin(), fitres.end(), &data[0], filter);
      const auto N{static_cast<Scalar>(data.size())};
      const auto median{select((static_cast<int>(data.size()) - 1) / 2, data)};
      const auto adev{(abs(data.array() - median).sum()) / N};
      const auto mean_stdev{adev / sqrt(N)};
      const auto stdev_stdev{adev / sqrt(2 * N)};
      return make_tuple(median, mean_stdev, adev, stdev_stdev);
    }};
    auto stat_mean{[&](const auto &fitres,
                       auto filter) -> tuple<Scalar, Scalar, Scalar, Scalar> {
      // compute mean and standard deviation Numerical Recipes 14.1.2 and 14.1.8
      auto data{Vec(fitres.size())};
      data.resize(fitres.size());
      transform(fitres.begin(), fitres.end(), &data[0], filter);
      const auto N{static_cast<Scalar>(data.size())};
      const auto mean{data.sum() / N};
      const auto stdev{sqrt(((pow(data.array() - mean, 2).sum()) -
                             (pow((data.array() - mean).sum(), 2) / N)) /
                            (N - 1.0F))};
      const auto mean_stdev{stdev / sqrt(N)};
      const auto stdev_stdev{stdev / sqrt(2 * N)};
      return make_tuple(mean, mean_stdev, stdev, stdev_stdev);
    }};
    auto stat{[&](const auto &fitres, auto filter) {
      if (meanOption->is_set()) {
        return stat_mean(fitres, filter);
      } else {
        return stat_median(fitres, filter);
      }
    }};
    auto generate_fit{[&]() {
      transform(&x[0], &x[0] + x.size(), &y[0],
                [&](Scalar xi) { return Sig * dis(gen) + A + B * xi; });
      return Fitab(x, y);
    }};
    auto fitres{vector<Fitab>()};
    fitres.reserve(repeat);
    generate_n(back_inserter(fitres), repeat, generate_fit);
    const auto a{stat(fitres, [&](const Fitab &f) { return f.a; })};
    const auto siga{stat(fitres, [&](const Fitab &f) { return f.siga; })};
    const auto b{stat(fitres, [&](const Fitab &f) { return f.b; })};
    const auto sigb{stat(fitres, [&](const Fitab &f) { return f.sigb; })};
    const auto chi2{stat(fitres, [&](const Fitab &f) { return f.chi2; })};
    const auto sigdat{stat(fitres, [&](const Fitab &f) { return f.sigdat; })};
    return make_tuple(a, siga, b, sigb, chi2, sigdat);
  }};
  for (decltype(0 + numberTrials + 1) i = 0; i < numberTrials; i += 1) {
    const auto dA{generatorDeltaIntercept};
    const auto A{generatorIntercept + dA * dis(gen)};
    const auto dB{generatorDeltaSlope};
    const auto B{generatorSlope + dB * dis(gen)};
    const auto Sig{generatorSigma};
    auto [a, siga, b, sigb, chi2,
          sigdat]{lin(numberPoints, A, B, Sig, numberRepeats)};
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
