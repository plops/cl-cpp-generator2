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
// dynamic rows, 1 column
using Vec = Matrix<Scalar, Dynamic, 1, 0, 8192, 1>;
// dynamic rows, 3 columns
using Mat = Matrix<Scalar, Dynamic, 3, 0, 8192, 1>;
using VecI = const Vec;
using VecO = Vec;
using MatI = const Mat;
class Fitmrq {
public:
  GaussianModel(Vec xx, Vec yy) : x{std::move(xx)}, y{std::move(yy)} {}
  Vec x, y, sig;
  constexpr int NDONE = 4, ITMAX = 1000;
  int ndat, ma, mfit;
  Scalar tol;
  void (*func)(const Scalar, VecI &, Scalar &, VecO &);
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
