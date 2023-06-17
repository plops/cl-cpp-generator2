#include <SFML/Graphics.hpp>
#include <armadillo>
#include <iostream>
#include <memory>

arma::vec compute_psi() {
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
  }
  std::cout << "Ground state energy: " << energy(0) << std::endl;

  return psi;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto psi = compute_psi();
  auto win =
      std::make_unique<sf::RenderWindow>(sf::VideoMode(800, 600), "psi plot");
  auto plot = sf::VertexArray(sf::LinesStrip, psi.n_elem);
  for (auto i = 0; i < psi.n_elem; i += 1) {
    auto x = float(i) / (psi.n_elem - 1) * win->(getSize().x);
    auto y = (1.0f - std::abs(psi(i))) * win->(getSize().y);
    plot[i].position = sf::Vector2f(x, y);
  }
  while (win->isOpen()) {
    auto event = sf::Event();
    while (win->pollEvent(event)) {
      if (sf::Event::Closed == event.type) {
        win->close();
      }
    }
    win->clear();
    win->draw(plot);
    win->display();
  }

  return EXIT_SUCCESS;
}
