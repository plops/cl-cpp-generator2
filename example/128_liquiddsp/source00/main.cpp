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
  auto mu = 5.00e-2f;
  auto alpha = 0.60f;
  auto w = vector<complex<float>>(wLen);
  auto b = vector<complex<float>>(bLen);
  auto xPrime = complex<float>(0.f);
  for (auto i = 0; i < wLen; i += 1) {
    w[i] = i == (wLen / 2) ? 1.0f : 0.f;
  }
  odtimes(n(numSymbols));

  return 0;
}
