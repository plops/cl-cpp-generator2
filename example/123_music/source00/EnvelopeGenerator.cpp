// no preamble
#include "EnvelopeGenerator.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
EnvelopeGenerator::EnvelopeGenerator(double sample_rate, double attack,
                                     double decay, double sustain,
                                     double release)
    : sample_rate_(sample_rate), attack_(attack), decay_(decay),
      sustain_(sustain), release_(release),
      current_state_(EnvelopeGeneratorState::Idle), current_amplitude_((0.)),
      attack_increment_((0.)), decay_increment_((0.)),
      release_increment_((0.)) {}
void EnvelopeGenerator::note_on() {
  current_state_ = EnvelopeGeneratorState::Attack;
  attack_increment_ = (((1.0)) / (((sample_rate_) * (attack_))));
}
void EnvelopeGenerator::note_off() {
  current_state_ = EnvelopeGeneratorState::Release;
  release_increment_ =
      ((((current_amplitude_) - ((0.)))) / (((sample_rate_) * (release_))));
}
double EnvelopeGenerator::next_amplitude() {
  switch (current_state_) {
  case EnvelopeGeneratorState::Attack: {
    (current_amplitude_) += (attack_increment_);
    if ((((1.0)) <= (current_amplitude_))) {
      current_amplitude_ = (1.0);
      current_state_ = EnvelopeGeneratorState::Decay;
      decay_increment_ =
          (((((1.0)) - (sustain_))) / (((sample_rate_) * (decay_))));
    }
    break;
  }
  case EnvelopeGeneratorState::Decay: {
    (current_amplitude_) -= (decay_increment_);
    if (((current_amplitude_) <= (sustain_))) {
      current_amplitude_ = sustain_;
      current_state_ = EnvelopeGeneratorState::Sustain;
    }
    break;
  }
  case EnvelopeGeneratorState::Sustain: {
    // amplitude remains constant

    break;
  }
  case EnvelopeGeneratorState::Release: {
    (current_amplitude_) -= (release_increment_);
    if (((current_amplitude_) <= ((0.)))) {
      current_amplitude_ = (0.);
      current_state_ = EnvelopeGeneratorState::Idle;
    }
    break;
  }
  case EnvelopeGeneratorState::Idle: {
    // amplitude remains zero

    break;
  }
  default: {
    // amplitude remains zero

    break;
  }
  }
  return current_amplitude_;
}
