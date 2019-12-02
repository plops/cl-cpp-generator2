
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <chrono>
#include <iostream>
#include <unordered_map>
State state = {};
int main() {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  state._filename =
      "/home/martin/Downloads/"
      "s1a-ew-raw-s-hv-20191130t152915-20191130t153018-030142-0371ab.dat";
  init_mmap(state._filename);
  init_collect_packet_headers();
  auto packet_idx = 0;
  std::unordered_map<int, int> map_ele;
  for (auto &e : state._header_data) {
    auto offset = state._header_offset[packet_idx];
    auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
    auto ele = ((0xF) & ((p[60]) >> (4)));
    auto number_of_quads =
        ((((0x1) * (p[66]))) + (((0x100) * (((0xFF) & (p[65]))))));
    (map_ele[ele]) += (number_of_quads);
    (packet_idx)++;
  };
  auto ma = (-1.e+0f);
  auto ma_ele = -1;
  for (auto &elevation : map_ele) {
    auto number_of_Mquads = ((elevation.second) / ((1.e+6f)));
    auto elevation_beam_address = elevation.first;
    if (ma < number_of_Mquads) {
      ma = number_of_Mquads;
      ma_ele = elevation_beam_address;
    };
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("map_ele") << (" ") << (std::setw(8))
                << (" elevation_beam_address=") << (elevation_beam_address)
                << (std::setw(8)) << (" number_of_Mquads=")
                << (number_of_Mquads) << (std::endl);
  };
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("largest ele") << (" ")
              << (std::setw(8)) << (" ma_ele=") << (ma_ele) << (std::setw(8))
              << (" ma=") << (ma) << (std::endl);
  {
    std::unordered_map<int, int> map_azi;
    auto packet_idx = 0;
    for (auto &e : state._header_data) {
      auto offset = state._header_offset[packet_idx];
      auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
      auto azi = ((((0x1) * (p[61]))) + (((0x100) * (((0x3) & (p[60]))))));
      auto number_of_quads =
          ((((0x1) * (p[66]))) + (((0x100) * (((0xFF) & (p[65]))))));
      (map_azi[azi]) += (number_of_quads);
      (packet_idx)++;
    };
    for (auto &azi : map_azi) {
      auto number_of_Mquads = ((azi.second) / ((1.e+6f)));
      auto azi_beam_address = azi.first;
      std::setprecision(3);
      (std::cout) << (std::setw(10))
                  << (((std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count()) -
                       (state._start_time)))
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" ") << ("map_azi") << (" ")
                  << (std::setw(8)) << (" azi_beam_address=")
                  << (azi_beam_address) << (std::setw(8))
                  << (" number_of_Mquads=") << (number_of_Mquads)
                  << (std::endl);
    };
  };
  std::array<std::complex<float>, 65535> output;
  auto n = init_decode_packet(0, output);
  destroy_mmap();
};