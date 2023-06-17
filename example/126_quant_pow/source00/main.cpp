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
  auto psi = arma::randu<arma::vec>(N);
  auto energy = arma::vec();
  auto status = arma::eigs_sym(energy, psi, H, 1, "sm");
  if (false == status) {
    std::cout << "Eigensolver failed." << energy << std::endl;
    return -1;
  }
  std::cout << "Ground state energy: " << energy(0) << std::endl;

  return 0;
}
