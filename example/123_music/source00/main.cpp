#include "EnvelopeGenerator.h"
#include "WavetableOscillator.h"
#include <vector>
#define FMT_HEADER_ONLY
#include "core.h"
#include <cmath>
#include <iostream>
#include <portaudio.h>

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
  {
    auto err = Pa_Initialize();
    if ((!((paNoError) == (err)))) {
      fmt::print("  Pa_GetErrorText(err)='{}'\n", Pa_GetErrorText(err));
      return 1;
    }
  }
  PaStream *stream = nullptr;
  auto userData = std::make_pair(&osc, &env);
  {
    auto err = Pa_OpenDefaultStream(&stream, 0, 2, paFloat32, sample_rate, 256,
                                    paCallback, &userData);
    if ((!((paNoError) == (err)))) {
      fmt::print("  Pa_GetErrorText(err)='{}'\n", Pa_GetErrorText(err));
      return 1;
    }
  }
  {
    auto err = Pa_StartStream(stream);
    if ((!((paNoError) == (err)))) {
      fmt::print("  Pa_GetErrorText(err)='{}'\n", Pa_GetErrorText(err));
      return 1;
    }
  }
  env.note_on();
  Pa_Sleep(1000);
  env.note_off();
  Pa_Sleep(2000);
  {
    auto err = Pa_StopStream(stream);
    if ((!((paNoError) == (err)))) {
      fmt::print("  Pa_GetErrorText(err)='{}'\n", Pa_GetErrorText(err));
      return 1;
    }
  }
  {
    auto err = Pa_CloseStream(stream);
    if ((!((paNoError) == (err)))) {
      fmt::print("  Pa_GetErrorText(err)='{}'\n", Pa_GetErrorText(err));
      return 1;
    }
  }
  Pa_Terminate();

  return 0;
}
