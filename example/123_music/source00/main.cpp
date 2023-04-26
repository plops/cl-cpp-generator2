#include "EnvelopeGenerator.h"
#include "WavetableOscillator.h"
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  auto sample_rate = (4.410e+4);
  auto wavetable_size = 1024U;
  auto wavetable = ([](auto size) {
    auto wavetable = std::vector<double>(size);
    for (auto i = 0; (i) < (size); (i) += (1)) {
      wavetable[i] =
          std::sin(((((2) * (M_PI) * (i))) / (static_cast<double>(size))));
    }
    return wavetable;
  })(wavetable_size);
  auto osc = WavetableOscillator(sample_rate, wavetable);
  osc.set_frequency((4.40e+2));
  auto attack = (1.00e-2);
  auto decay = (0.200000000000000000000000000000);
  auto sustain = (0.600000000000000000000000000000);
  auto release = (0.50);
  auto env = EnvelopeGenerator(sample_rate, attack, decay, sustain, release);
  env.note_on();
  auto count = 0;
  for (auto i = 0; (i) < (2000); (i) += (1)) {
    auto osc_output = osc.next_sample();
    auto env_amplitude = env.next_amplitude();
    auto output_sample = ((osc_output) * (env_amplitude));
    std::cout << count << (" ") << output_sample << std::endl;
    (count)++;
  }
  env.note_off();
  for (auto i = 0; (i) < (22050); (i) += (1)) {
    auto osc_output = osc.next_sample();
    auto env_amplitude = env.next_amplitude();
    auto output_sample = ((osc_output) * (env_amplitude));
    std::cout << count << (" ") << output_sample << std::endl;
    (count)++;
  }

  return 0;
}
