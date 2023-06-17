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
      // subdiagonal
      H(i, i - 1) = (-1.0) / (dx * dx);
    }
    // main diagonal

    H(i, i) = 2.0 / (dx * dx);

    if (i < (N - 1)) {
      // superdiagonal
      H(i, i + 1) = (-1.0) / (dx * dx);
    }
  }
  // Initialize a random vector
  auto psi = arma::randu<arma::vec>(N);
  // Normalize psi
  psi /= arma::norm(psi);
  auto energy = arma::vec();
  auto status = arma::eigs_sym(energy, psi, H, 1, "sm");
  if (false == status) {
    std::cout << "Eigensolver failed." << energy << std::endl;
    return -1;
  }
  std::cout << "Ground state energy: " << energy(0) << std::endl;

  return 0;
}
