#include <armadillo>
#include <iostream>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  // N .. Number of discretizatino points
  // L .. Size of the box
  // dx .. Grid spacing

  auto N = 1000;
  auto L = 1.0;
  auto dx = L / (N + 1);
  auto H = arma::sp_mat(N, N);
  for (auto i = 0; i < N; i += 1) {
    if (0 < i) {
      H(i, i - 1) = (-1.0) / (dx * dx);
    }
    H(i, i) = 2.0 / (dx * dx);

    if (i < (N - 1)) {
      H(i, i + 1) = (-1.0) / (dx * dx);
    }
  }
  auto psi = arma::randu<arma::vec>(N);
  for (auto iter = 0; iter < 10000; iter += 1) {
    psi = (H * psi);

    psi /= arma::norm(psi);
  }
  auto energy = arma::dot(psi, H * psi);
  std::cout << ("Ground state energy: ") << energy << std::endl;

  return 0;
}
