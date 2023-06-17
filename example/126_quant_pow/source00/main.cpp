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

  return 0;
}
