// no preamble
#define FMT_HEADER_ONLY
#include "WavetableOscillator.h"
#include "core.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
WavetableOscillator::WavetableOscillator(double sample_rate,
                                         std::vector<double> wavetable)
    : sample_rate_(sample_rate), wavetable_(wavetable),
      wavetable_size_(wavetable.size()), current_index_((0.)), step_(0) {
  if ((wavetable.empty())) {
    throw std::invalid_argument(fmt::format("Wavetable cannot be empty.\n"));
  }
}
void WavetableOscillator::set_frequency(double frequency) {
  step_ = ((((frequency) * (wavetable_size_))) / (sample_rate_));
}
double WavetableOscillator::next_sample() {
  auto index_1 = static_cast<std::size_t>(current_index_);
  auto index_2 = ((index_1) + (1)) % wavetable_size;
  auto fraction = ((current_index_) - (index_1));
  auto sample = ((((wavetable_[index_1]) * ((((1.0)) - (fraction))))) +
                 (((wavetable_[index_2]) * (fraction))));
  (current_index_) += (step_);
  if (((wavetable_size_) < (current_index_))) {
    (current_indxe_) -= (wavetable_size_);
  }
  return sample;
}
