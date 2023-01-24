#include <Eigen/Dense>
using namespace Eigen;

double objective(const VectorXd &x) {
  auto y = VectorXd(3);
  y << 1, 2, 3;
  auto f = Vector3d(
      {(x(0) + (x(1) * y(0))), (x(0) + (x(1) * y(1))), (x(0) + (x(1) * y(2)))});
  auto err = ((y) - (f));

  return err.squaredNorm();
}

VectorXd gradient(const VectorXd &x) {
  auto y = VectorXd(3);
  y << 1, 2, 3;
  auto f = Vector3d(
      {(x(0) + (x(1) * y(0))), (x(0) + (x(1) * y(1))), (x(0) + (x(1) * y(2)))});
  auto err = ((y) - (f));

  auto result = VectorXd(2);
  result(0) = (-2 * err.sum());
  result(1) = (-2 * (err.transpose() * y).sum());

  return result;
}

MatrixXd hessian(const VectorXd &x) {
  auto y = VectorXd(3);
  y << 1, 2, 3;
  auto f = Vector3d(
      {(x(0) + (x(1) * y(0))), (x(0) + (x(1) * y(1))), (x(0) + (x(1) * y(2)))});
  auto err = ((y) - (f));

  auto result = MatrixXd(2, 2);
  result(0, 0) = (2 * y.size());
  result(0, 1) = (2 * y.sum());
  result(1, 0) = (2 * y.sum());
  result(1, 1) = (2 * (y.array() * y.array()).sum());

  return result;
}

int main(int argc, char **argv) {
  auto lam = (0.100000000000000000000000000000);
  auto x = VectorXd(2);
  auto tolerance = (1.00e-4f);
  auto maxIterations = 100;
  x << 1, 1;
  for (auto i = 0; i < maxIterations; i += 1) {
    auto grad = gradient(x);
    auto hess = hessian(x);
    hess += (lam * MatrixXd::Identity(x.size(), x.size()));
    auto step = -hess.ldlt().solve(grad);
    x += step;
    auto change = step.norm();
    if (change < tolerance) {
      break;
    }
    lam *= ((0.90f));
  }
}
