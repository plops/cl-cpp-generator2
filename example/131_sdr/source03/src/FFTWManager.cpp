// no preamble

#include <stdexcept>

#include "FFTWManager.h"
FFTWManager::FFTWManager(int window_size) : window_size_(window_size) {
  if (window_size_ <= 0) {
    throw std::invalid_argument("window size must be positive");
  }
  auto iter = plans.find(window_size_);
}
FFTWManager::~FFTWManager() {
  for (const auto &kv : plans) {
    fftw_destroy_plan(kv.second);
  }
}
