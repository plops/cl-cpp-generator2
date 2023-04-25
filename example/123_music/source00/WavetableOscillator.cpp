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
