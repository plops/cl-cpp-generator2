#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto M = 1 << 3;
  auto numSymbols = 1200;
  auto wLen = 13;
  auto mu = 5.00e-2F;
  auto alpha = 0.60F;
  auto w = vector<complex<float>>(wLen);
  auto b = vector<complex<float>>(wLen);
  auto xPrime = complex<float>(0.F);
  // Initialize arrays

  for (auto i = 0; i < wLen; i += 1) {
    w[i] = i == (wLen / 2) ? 1.0F : 0.F;
  }
  auto bufIndex = 0;
  for (auto n = 0; n < numSymbols; n += 1) {
    // x .. random transmitted phase-shift keying symbol
    // y .. computed received signal to be stored in buffer b

    auto x = exp(complex<float>(0, 1) * 2.0F * static_cast<float>(M_PI) *
                 (static_cast<float>(rand() % M) / M));
    auto y = (sqrt(1 - alpha) * x) + alpha + xPrime;
    xPrime = y;
    b[bufIndex] = y;
    bufIndex = ((bufIndex + 1) % wLen);

    // compute equalizer output r

    auto r = complex<float>(0.F);
    for (auto i = 0; i < wLen; i += 1) {
      r += b[((bufIndex + i) % wLen)] * conj(w[i]);
    }

    // compute expected signal (blind), skip first wLen symbols

    auto e = (n < wLen) ? complex<float>(0.F) : r - (r / abs(r));
    for (auto i = 0; i < wLen; i += 1) {
      w[i] -= mu * conj(e) * b[((bufIndex + i) % wLen)];
    }
    std::cout << ""
              << " y='" << y << "' "
              << " r='" << r << "' " << std::endl;
  }

  return 0;
}
