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

Scalar select(const int k, Vec &arr) {
  // Numerical Recipes 8.5: select a random partitioning element `a`  and
  // iterate through array.
  // move smaller elements to the left and larger to the right. (this is like
  // quicksort) sentinels at either end of the subarray reduce work in the inner
  // loop. leftmost sentienel is <= a, rightmost sentinel is>=a
  const auto n{static_cast<int>(arr.size())};
  Scalar a;
  auto ir{n - 1};
  auto l{0};
  auto i{0};
  auto j{0};
  auto mid{0};
  while (true) {
    if (ir <= l + 1) {
      // Active partition contains 1 or 2 elements
      if (ir == l + 1 && arr[ir] < arr[l]) {
        // Case of two elements
        std::swap(arr[l], arr[ir]);
      }
      return arr[k];
    } else {
      // Choose median of left, center and right elements as partitioning
      // element a
      // Also rearrange so that arr[l] <= arr[l+1], arr[ir]>=arr[l+1]
      mid = (l + ir) >> 1;
      std::swap(arr[mid], arr[(l + i)]);
      if (arr[ir] < arr[l]) {
        std::swap(arr[ir], arr[l]);
      }
      if (arr[ir] < arr[(l + 1)]) {
        std::swap(arr[ir], arr[(l + 1)]);
      }
      if (arr[(l + 1)] < arr[l]) {
        std::swap(arr[(l + 1)], arr[l]);
      }
      // Initialize pointers for partitioning
      i = l + 1;
      j = ir;
      a = arr[(l + 1)];
      // Inner loop
      while (true) {
        // Scan up to find element > a
        do {
          i++;
        } while (arr[i] < a);
        // Scan down to find element < a
        do {
          j--;
        } while (a < arr[j]);
        if (j < i) {
          // Pointers crossed. Partitioning complete
          break;
        }
        // Insert partitioning element
        std::swap(arr[i], arr[j]);
      }
      // Insert partitioning element
      arr[(l + 1)] = arr[j];
      arr[j] = a;
      // Keep active the partition that contains the kth element
      if (k <= j) {
        ir = (j - 1);
      }
      if (j <= k) {
        l = i;
      }
    }
  }
}

int main(int argc, char **argv) {
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
    auto b{stat_median(fitres, [&](const Fitab &f) { return f.b; })};
    auto siga{stat_median(fitres, [&](const Fitab &f) { return f.siga; })};
    auto sigb{stat_median(fitres, [&](const Fitab &f) { return f.sigb; })};
    auto chi2{stat_median(fitres, [&](const Fitab &f) { return f.chi2; })};
    auto sigdat{stat_median(fitres, [&](const Fitab &f) { return f.sigdat; })};
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
