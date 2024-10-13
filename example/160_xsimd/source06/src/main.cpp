#include "xsimd/xsimd.hpp"
#include <algorithm>
#include <cmath>
#include <execution>
#include <format>
#include <iostream>
#include <numeric>
#include <popl.hpp>
#include <random>
#include <thread>
#include <tuple>
#include <vector>
using namespace std;
using namespace xsimd;
using Scalar = float;
using Vec = std::vector<Scalar, xsimd::default_allocator<Scalar>>;
using VecI = const Vec;
using Batch = xsimd::batch<Scalar, avx2>;
constexpr auto Pol = std::execution::par_unseq;
// = sum(arr)
auto sum{[](VecI &arr) -> Scalar {
  const auto inc{Batch::size};
  const auto size{arr.size()};
  const auto vec_size{size - (size % inc)};
  auto sum{0.F};
  for (decltype(0 + vec_size + 1) i = 0; i < vec_size; i += inc) {
    sum += reduce_add(Batch::load_aligned(&arr[i]));
  }
  return sum;
}};
// res = arr - s .. array subtract
auto asub{[](Vec &res, VecI &arr, Scalar s) {
  const auto inc{Batch::size};
  const auto size{arr.size()};
  const auto vec_size{size - (size % inc)};
  for (decltype(0 + vec_size + 1) i = 0; i < vec_size; i += inc) {
    const auto a{Batch::load_aligned(&arr[i]) - s};
    a.store_aligned(&res[i]);
  }
}};
// = sum(| arr - s |) .. subtract scalar, compute absolute value, sum
auto subabssum{[](VecI &arr, Scalar s) -> Scalar {
  const auto inc{Batch::size};
  const auto size{arr.size()};
  const auto vec_size{size - (size % inc)};
  auto sum{0.F};
  for (decltype(0 + vec_size + 1) i = 0; i < vec_size; i += inc) {
    sum += reduce_add(abs(Batch::load_aligned(&arr[i]) - s));
  }
  return sum;
}};
// = sum(arr**2)
auto squaredsum{[](VecI &arr) -> Scalar {
  const auto inc{Batch::size};
  const auto size{arr.size()};
  const auto vec_size{size - (size % inc)};
  auto sum{0.F};
  for (decltype(0 + vec_size + 1) i = 0; i < vec_size; i += inc) {
    sum += reduce_add(pow(Batch::load_aligned(&arr[i]), 2));
  }
  return sum;
}};
// = sum( tt*y / st2 )
auto compute_b{[](VecI &tt, VecI &y, Scalar st2) -> Scalar {
  const auto inc{Batch::size};
  const auto size{tt.size()};
  const auto vec_size{size - (size % inc)};
  auto sum{0.F};
  for (decltype(0 + vec_size + 1) i = 0; i < vec_size; i += inc) {
    auto att{Batch::load_aligned(&tt[i])};
    auto ay{Batch::load_aligned(&y[i])};
    sum += reduce_add((att * ay) / st2);
  }
  return sum;
}};
// = sum((y-a-b*x)**2)
auto compute_chi2{[](VecI &y, Scalar a, Scalar b, VecI &x) -> Scalar {
  const auto inc{Batch::size};
  const auto size{y.size()};
  const auto vec_size{size - (size % inc)};
  auto sum{0.F};
  for (decltype(0 + vec_size + 1) i = 0; i < vec_size; i += inc) {
    auto ax{Batch::load_aligned(&x[i])};
    auto ay{Batch::load_aligned(&y[i])};
    sum += reduce_add(pow(ay - a - (ax * b), 2));
  }
  return sum;
}};
// From Numerical Recipes
class Fitab {
public:
  Fitab(VecI &xx, VecI &yy) : ndata{static_cast<int>(xx.size())}, x{xx}, y{yy} {
    const auto inc{Batch::size};
    const auto size{x.size()};
    const auto vec_size{size - (size % inc)};
    const auto sx{sum(x)};
    const auto sy{sum(y)};
    const auto ss{static_cast<Scalar>(ndata)};
    const auto sxoss{sx / ss};
    const auto tt{([&]() {
      auto res{Vec(size)};
      asub(res, x, sxoss);
      return res;
    })()};
    const auto st2{squaredsum(tt)};
    b = compute_b(tt, y, st2);
    a = ((sy - (b * sx)) / ss);
    siga = sqrt((1.0F + ((sx * sx) / (ss * st2))) / ss);
    sigb = sqrt(1.0F / st2);
    chi2 = compute_chi2(y, a, b, x);
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

//  = sum( (data-s) ** 2 )
auto var_pass1{[](VecI &arr, Scalar s) -> Scalar {
  const auto inc{Batch::size};
  const auto size{arr.size()};
  const auto vec_size{size - (size % inc)};
  auto sum{0.F};
  for (decltype(0 + vec_size + 1) i = 0; i < vec_size; i += inc) {
    sum += reduce_add(pow(Batch::load_aligned(&arr[i]) - s, 2));
  }
  return sum;
}};
//  = sum( data-s )
auto var_pass2{[](VecI &arr, Scalar s) -> Scalar {
  const auto inc{Batch::size};
  const auto size{arr.size()};
  const auto vec_size{size - (size % inc)};
  auto sum{0.F};
  for (decltype(0 + vec_size + 1) i = 0; i < vec_size; i += inc) {
    sum += reduce_add(Batch::load_aligned(&arr[i]) - s);
  }
  return sum;
}};
auto stat_mean{[](const auto &fitres,
                  auto filter) -> tuple<Scalar, Scalar, Scalar, Scalar> {
  // compute mean and standard deviation Numerical Recipes 14.1.2 and 14.1.8
  auto data{Vec(fitres.size())};
  data.resize(fitres.size());
  transform(fitres.begin(), fitres.end(), &data[0], filter);
  const auto N{static_cast<Scalar>(data.size())};
  const auto mean{sum(data) / N};
  const auto stdev{
      sqrt((var_pass1(data, mean) - (pow(var_pass2(data, mean), 2.0F) / N)) /
           (N - 1.0F))};
  const auto mean_stdev{stdev / sqrt(N)};
  const auto stdev_stdev{stdev / sqrt(2 * N)};
  return make_tuple(mean, mean_stdev, stdev, stdev_stdev);
}};
auto stat{[](const auto &fitres, auto filter, auto meanOption) {
  return stat_mean(fitres, filter);
}};

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
  auto lin{[](auto n, auto A, auto B, auto Sig, auto repeat, auto &meanOption,
              auto &dis, auto &gen) {
    // number points must be divisible by 8 (avx2 batch size)
    assert((n % Batch::size) == 0);
    const auto x{([&]() {
      auto x{Vec(n)};
      iota(&x[0], &x[0] + x.size(), 1.0F);
      return x;
    })()};
    auto y{Vec(n)};
    auto generate_fit{[&x, &y, Sig, &dis, &gen, A, B]() {
      transform(&x[0], &x[0] + x.size(), &y[0],
                [Sig, &dis, &gen, A, B](Scalar xi) {
                  return Sig * dis(gen) + A + B * xi;
                });
      const auto f{Fitab(x, y)};
      const auto a{f.a};
      const auto siga{f.siga};
      const auto b{f.b};
      const auto sigb{f.sigb};
      const auto chi2{f.chi2};
      const auto sigdat{f.sigdat};
      return make_tuple(a, siga, b, sigb, chi2, sigdat);
    }};
    auto fitres{
        vector<tuple<Scalar, Scalar, Scalar, Scalar, Scalar, Scalar>>(repeat)};
    const auto numThreads{8};
    const auto elements_per_thread{repeat / numThreads};
    auto threads{vector<jthread>(numThreads)};
    for (decltype(0 + numThreads + 1) j = 0; j < numThreads; j += 1) {
      threads[j] = jthread([elements_per_thread, j, &fitres, &generate_fit]() {
        for (decltype(0 + elements_per_thread + 1) i = 0;
             i < elements_per_thread; i += 1) {
          fitres[(j * elements_per_thread + i)] = generate_fit();
        }
      });
    }
    for (auto &&th : threads) {
      if (th.joinable()) {
        th.join();
      }
    }
    const auto a{
        stat(fitres, [&](const auto &f) { return get<0>(f); }, meanOption)};
    const auto siga{
        stat(fitres, [&](const auto &f) { return get<1>(f); }, meanOption)};
    const auto b{
        stat(fitres, [&](const auto &f) { return get<2>(f); }, meanOption)};
    const auto sigb{
        stat(fitres, [&](const auto &f) { return get<3>(f); }, meanOption)};
    const auto chi2{
        stat(fitres, [&](const auto &f) { return get<4>(f); }, meanOption)};
    const auto sigdat{
        stat(fitres, [&](const auto &f) { return get<5>(f); }, meanOption)};
    return make_tuple(a, siga, b, sigb, chi2, sigdat);
  }};
  for (decltype(0 + numberTrials + 1) i = 0; i < numberTrials; i += 1) {
    const auto dA{generatorDeltaIntercept};
    const auto A{generatorIntercept + dA * dis(gen)};
    const auto dB{generatorDeltaSlope};
    const auto B{generatorSlope + dB * dis(gen)};
    const auto Sig{generatorSigma};
    auto [a, siga, b, sigb, chi2, sigdat]{
        lin(numberPoints, A, B, Sig, numberRepeats, meanOption, dis, gen)};
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
