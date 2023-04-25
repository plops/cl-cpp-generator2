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
  for (auto i = 0; (i) < (100); (i) += (1)) {
    std::cout << (osc.next_sample()) << std::endl;
  }

  return 0;
}
