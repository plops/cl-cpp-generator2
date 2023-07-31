// no preamble

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "GpsCACodeGenerator.h"
GpsCACodeGenerator::GpsCACodeGenerator(int prn) : prn_(prn) {
  if (prn_ < 1 || 32 < prn_) {
    throw std::invalid_argument("Invalid PRN: " + std::to_string(prn_));
  }
  g1_.resize(register_size_, true);
  g2_.resize(register_size_, true);
}
std::vector<bool> GpsCACodeGenerator::generate_sequence(size_t n) {
  auto sequence = std::vector<bool>();
  sequence.reserve(n);
  for (size_t i = 0; i < n; i += 1) {
    sequence.push_back(step());
  }
  return sequence;
}
void GpsCACodeGenerator::print_square(const std::vector<bool> &v) {
  auto size = v.size();
  auto side = static_cast<size_t>(std::sqrt(size));
  for (size_t i = 0; i < size; i += 1) {
    auto o = v[i] ? "\u2588" : " ";
    std::cout << o;
    if (0 == ((i + 1) % side)) {
      std::cout << "\n";
    }
  }
}
bool GpsCACodeGenerator::step() {
  auto new_g1_bit = false;
  for (const auto &i : g1_feedback_bits_) {
    new_g1_bit = new_g1_bit ^ g1_[(i - 1)];
  }

  auto new_g2_bit = false;
  for (const auto &i : g2_feedback_bits_) {
    new_g2_bit = new_g2_bit ^ g2_[(i - 1)];
  }

  g1_.push_front(new_g1_bit);
  g1_.pop_back();

  g2_.push_front(new_g2_bit);
  g2_.pop_back();

  auto delay1 = g2_shifts_[(prn_ - 1)].first;
  auto delay2 = g2_shifts_[(prn_ - 1)].second;

  return g1_.back() ^ g2_[(delay1 - 1)] ^ g2_[(delay2 - 1)];
}
