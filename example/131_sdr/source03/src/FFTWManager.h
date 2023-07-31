#ifndef FFTWMANAGER_H
#define FFTWMANAGER_H

#include <complex>
#include <fftw3.h>
#include <map>
#include <vector>

class FFTWManager {
public:
  explicit FFTWManager();
  fftw_plan get_plan(int windowSize, int nThreads = 1);
  std::vector<std::complex<double>>
  fftshift(const std::vector<std::complex<double>> &in);
  std::vector<std::complex<double>>
  fft(const std::vector<std::complex<double>> &in, int windowSize);
  ~FFTWManager();

private:
};

#endif /* !FFTWMANAGER_H */