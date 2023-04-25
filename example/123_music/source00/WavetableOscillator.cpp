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
    : sample_rate(sample_rate), wavetable(wavetable),
      wavetable_size(wavetable.size()), current_index((0.f)), step(0) {}
