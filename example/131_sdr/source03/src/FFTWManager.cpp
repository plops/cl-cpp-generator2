// no preamble

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "FFTWManager.h"
FFTWManager::FFTWManager() {}
fftw_plan FFTWManager::get_plan(int windowSize, int nThreads) {
  if (windowSize <= 0) {
    throw std::invalid_argument("window size must be positive");
  }
  std::cout << "lookup plan"
            << " windowSize='" << windowSize << "' "
            << " nThreads='" << nThreads << "' " << std::endl
            << std::flush;
  if (true) {
    auto wisdom_filename = "wisdom_" + std::to_string(windowSize) + ".wis";

    if (1 < nThreads) {
      wisdom_filename = "wisdom_" + std::to_string(windowSize) + "_threads" +
                        std::to_string(nThreads) + ".wis";
    }

    auto *in = fftw_alloc_complex(windowSize);
    auto *out = fftw_alloc_complex(windowSize);
    if (!in || !out) {
      fftw_free(in);
      fftw_free(out);
      throw std::runtime_error("Failed to allocate memory for fftw plan");
    }
    auto wisdomFile = std::ifstream(wisdom_filename);
    if (wisdomFile.good()) {
      std::cout << "read wisdom from existing file"
                << " wisdom_filename='" << wisdom_filename << "' " << std::endl
                << std::flush;
      wisdomFile.close();
      fftw_import_wisdom_from_filename(wisdom_filename.c_str());
    }
    if (1 < nThreads) {
      std::cout << "plan 1d fft with threads"
                << " nThreads='" << nThreads << "' " << std::endl
                << std::flush;
      fftw_plan_with_nthreads(nThreads);

    } else {
      std::cout << "plan 1d fft without threads"
                << " nThreads='" << nThreads << "' " << std::endl
                << std::flush;
    }
    auto p = fftw_plan_dft_1d(windowSize, in, out, FFTW_FORWARD, FFTW_MEASURE);

    if (!p) {
      fftw_free(in);
      fftw_free(out);
      throw std::runtime_error("Failed to create fftw plan");
    }
    std::cout << "plan successfully created" << std::endl << std::flush;
    if (!wisdomFile.good()) {
      std::cout << "store wisdom to file"
                << " wisdom_filename='" << wisdom_filename << "' " << std::endl
                << std::flush;
      wisdomFile.close();
      fftw_export_wisdom_to_filename(wisdom_filename.c_str());
    }

    std::cout << "free in and out" << std::endl << std::flush;
    fftw_free(in);
    fftw_free(out);
    return p;
  }
}
std::vector<std::complex<double>>
FFTWManager::fftshift(const std::vector<std::complex<double>> &in) {
  auto mid = in.begin() + (in.size() / 2);
  auto out = std::vector<std::complex<double>>(in.size());
  std::copy(mid, in.end(), out.begin());
  std::copy(in.begin(), mid, out.begin() + std::distance(mid, in.end()));
  return out;
}
std::vector<std::complex<double>>
FFTWManager::fft(const std::vector<std::complex<double>> &in, int windowSize) {
  if (windowSize != in.size()) {
    throw std::invalid_argument("Input size must match window size.");
  }
  auto out = std::vector<std::complex<double>>(windowSize);
  fftw_execute_dft(get_plan(windowSize, 6),
                   reinterpret_cast<fftw_complex *>(
                       const_cast<std::complex<double> *>(in.data())),
                   reinterpret_cast<fftw_complex *>(
                       const_cast<std::complex<double> *>(out.data())));
  return fftshift(out);
}
FFTWManager::~FFTWManager() {}
