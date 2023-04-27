#include "EnvelopeGenerator.h"
#include "WavetableOscillator.h"
#include <cmath>
#include <iostream>
#include <portaudio.h>
#include <vector>

static int paCallback(const void *input_buffer, void *output_buffer,
                      unsigned long frames_per_buffer,
                      const PaStreamCallbackTimeInfo *time_info,
                      PaStreamCallbackFlags status_flags, void *user_data) {
  auto *data =
      static_cast<std::pair<WavetableOscillator *, EnvelopeGenerator *> *>(
          user_data);
  auto *osc = data->first;
  auto *env = data->second;
  auto *out = static_cast<float *>(output_buffer);
  for (auto i = 0; (i) < (frames_per_buffer); (i) += (1)) {
    auto osc_ = osc->next_sample();
    auto env_ = env->next_amplitude();
    auto out_ = ((osc_) * (env_));
    // left and right channel
    *out++ = static_cast<float>(out_);
    *out++ = static_cast<float>(out_);
  }
  return paContinue;
}

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
  auto err = Pa_Initialize();

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
