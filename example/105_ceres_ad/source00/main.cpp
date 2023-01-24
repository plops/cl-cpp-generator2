#include <Eigen/Dense>
using namespace Eigen;

double objective(const VectorXd &x) {
  auto y = VectorXd(3);
  y << 1, 2, 3;
  auto f = (x(0) + (x(1) * y));
  auto err = ((y) - (f));

  return err.squaredNorm();
}

int main(int argc, char **argv) {}
