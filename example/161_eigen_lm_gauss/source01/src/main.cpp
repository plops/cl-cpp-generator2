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
  return amplitude * exp((x0 * x0) / (-2.0F * sigma * sigma));
}

class GaussianModel {
public:
  GaussianModel(VecI &xx, VecI &yy) : x{xx}, y{yy} {}
  void operator()(VecI &parameters, Vec &residuals) {
    const auto amplitude{parameters(0)};
    const auto mean{parameters(1)};
    const auto sigma{parameters(2)};
    const auto gaussian_values{
        amplitude *
        exp((((x.array()) - mean).array().square()) / (-2.0F * sigma * sigma))};
    residuals = -gaussian_values.array() + y;
  }
  void jacobian(VecI &parameters, Mat &jac) {
    const auto amplitude{parameters(0)};
    const auto mean{parameters(1)};
    const auto sigma{parameters(2)};
    const auto gaussian_values{
        amplitude *
        exp((((x.array()) - mean).array().square()) / (-2.0F * sigma * sigma))};
    // Derivative with respect to amplitude
    jac.col(0) = gaussian_values;
    const auto diff_x{(x.array()) - mean};
    const auto exp_term_matrix{
        exp((x.array().square().colwise()) / (-2.0F * sigma * sigma)).matrix()};
    const auto denominator{sigma * sigma};
    // Derivative with respect to mean
    jac.col(1) = diff_x.array().rowwise() * exp_term_matrix.rowwise() *
                 (amplitude / denominator);
    // Derivative with respect to sigma
    jac.col(2) = diff_x.array().square().rowwise() * exp_term_matrix.rowwise() *
                 (amplitude / (denominator * sigma * sigma));
  }
  Vec x, y;
};

Vec &lm(const GaussianModel &model, VecI &initial_guess, Scalar lambda) {
  const auto maxIter{100};
  const auto tolerance{1.00e-4F};
  auto parameters{initial_guess};
  auto residuals{Vec(model.x.size())};
  auto jacobian{Mat(model.x.size(), 3)};
  for (decltype(0 + maxIter + 1) iter = 0; iter < maxIter; iter += 1) {
    model(residuals, model.jacobian(parameters, jacobian));
    auto residual_norm{residuals.norm()};
    if (residual_norm < tolerance) {
      break;
    }
    auto jTj{jacobian.transpose() * jacobian +
             lambda * MatrixXd::Identity(3, 3)};
    auto delta{-jacobian.transpose() * residuals};
    auto parameters_new{parameters + delta};
    auto residual_norm_new{
        model(parameters_new, model.jacobian(parameters_new, jacobian)).norm()};
    if (residual_norm_new < residual_norm) {
      parameters = parameters_new;
      lambda /= 10.F;
    } else {
      lambda *= 10.F;
    }
  }
  return parameters;
}

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
  auto x{Vec(-2(-1.50F, -1, -0.50F, 0, 0.50F, 1, 1.50F, 2))};
  auto y{Vec(6.740e-2F(0.13580F, 0.28650F, 0.49330F, 1, 0.49330F, 0.28650F,
                       0.13580F, 6.740e-2F))};
  auto initial_guess{Vec(1.0F, 0.F, 1.0F)};
  auto lamb{0.10F};
  auto parameters{lm(GaussianModel(x, y), initial_guess, lamb)};
  return 0;
}
