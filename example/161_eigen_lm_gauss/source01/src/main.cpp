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
using Vec = Matrix<Scalar, Dynamic, 1, 0, 8192, 1>;
using Mat = Matrix<Scalar, Dynamic, 3, 0, 8192, 1>;
using VecI = const Vec;
using MatI = const Mat;

Scalar gaussian(Scalar x, Scalar amplitude, Scalar mean, Scalar sigma) {
  auto x0{x - mean};
  return amplitude * exp((x0 * x0) / (-2.0F * sigam * sigma));
}

class GaussianModel {
public:
  GaussianModel(VecI &xx, VecI &yy) : x{xx}, y{yy} {}
  void operator()(VecI &parameters, Vec &residuals) {
    const auto amplitude{parameters(0)};
    const auto mean{parameters(1)};
    const auto sigma{parameters(2)};
    const auto guassian_values{
        amplitude *
        exp(exp((x - mean, array(), square())) / (-2.0F * sigma * sigma))};
    residuals = (y - gaussian_values);
  }
  void jacobian(VecI &parameters, Mat &jac) {
    const auto amplitude{parameters(0)};
    const auto mean{parameters(1)};
    const auto sigma{parameters(2)};
    const auto guassian_values{
        amplitude *
        exp(exp((x - mean, array(), square())) / (-2.0F * sigma * sigma))};
    // Derivative with respect to amplitude
    jac.col(0) = gaussian_values;
    const auto diff_x{x - mean};
    const auto exp_term_matrix{
        exp((x.array().square().colwise()) / (-2.0F * sigma * sigm)).matrix()};
    const auto denominator{sigma * sigma};
    // Derivative with respect to mean
    jac.col(1) = amplitude * diff_x.array().rowwise() *
                 (exp_term_matrix.rowwise() / denominator);
    // Derivative with respect to sigma
    jac.col(2) = amplitude * diff_x.array().square().rowwise() *
                 (exp_term_matrix.rowwise() / (denominator * sigma * sigma));
  }
  Vec x, y;
};

int main(int argc, char **argv) {
  auto op{popl::OptionParser("allowed options")};
  auto numberPoints{int(1024)};
  auto helpOption{op.add<popl::Switch>("h", "help", "produce help message")};
  auto verboseOption{
      op.add<popl::Switch>("v", "verbose", "produce verbose output")};
  auto meanOption{op.add<popl::Switch>(
      "m", "mean",
      "Print mean and standard deviation statistics, otherwise print median "
      "and mean absolute deviation from it")};
  auto numberPointsOption{op.add<popl::Value<int>>(
      "p", "numberPoints", "parameter", 1024, &numberPoints)};
  op.parse(argc, argv);
  if (helpOption->is_set()) {
    cout << op << endl;
    exit(0);
  }
  std::cout << std::format("(:thread::hardware_concurrency() '{}')\n",
                           thread::hardware_concurrency());
  auto gen{mt19937(random_device{}())};
  auto dis{normal_distribution<float>(0.F, 1.0F)};
  return 0;
}
