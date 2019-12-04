
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <cassert>
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
  auto mi_data_delay = 10000000;
  auto ma_data_delay = -1;
  auto ma_data_end = -1;
  auto ele_number_echoes = 0;
  {
    std::unordered_map<int, int> map_azi;
    auto packet_idx = 0;
    for (auto &e : state._header_data) {
      auto offset = state._header_offset[packet_idx];
      auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
      auto ele = ((0xF) & ((p[60]) >> (4)));
      auto azi = ((((0x1) * (p[61]))) + (((0x100) * (((0x3) & (p[60]))))));
      auto number_of_quads =
          ((((0x1) * (p[66]))) + (((0x100) * (((0xFF) & (p[65]))))));
      auto data_delay = ((40) + (((((0x1) * (p[55]))) + (((0x100) * (p[54]))) +
                                  (((0x10000) * (((0xFF) & (p[53]))))))));
      if ((ele) == (ma_ele)) {
        (ele_number_echoes)++;
        if (data_delay < mi_data_delay) {
          mi_data_delay = data_delay;
        };
        if (ma_data_delay < data_delay) {
          ma_data_delay = data_delay;
        };
        auto v = ((data_delay) + (((2) * (number_of_quads))));
        if (ma_data_end < v) {
          ma_data_end = v;
        };
        (map_azi[azi]) += (number_of_quads);
      };
      (packet_idx)++;
    };
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("data_delay") << (" ")
                << (std::setw(8)) << (" mi_data_delay=") << (mi_data_delay)
                << (std::setw(8)) << (" ma_data_delay=") << (ma_data_delay)
                << (std::setw(8)) << (" ma_data_end=") << (ma_data_end)
                << (std::setw(8)) << (" ele_number_echoes=")
                << (ele_number_echoes) << (std::endl);
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
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("start big allocation") << (" ")
              << (std::endl);
  auto n0 = ((ma_data_end) + (((ma_data_delay) - (mi_data_delay))));
  auto sar_image = new std::complex<float>[((n0) * (ele_number_echoes))];
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("end big allocation") << (" ")
              << (std::setw(8)) << (" (((1.e-6f))*(n0)*(ele_number_echoes))=")
              << ((((1.e-6f)) * (n0) * (ele_number_echoes))) << (std::endl);
  {
    auto packet_idx = 0;
    auto ele_count = 0;
    for (auto &e : state._header_data) {
      auto offset = state._header_offset[packet_idx];
      auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
      auto ele = ((0xF) & ((p[60]) >> (4)));
      auto number_of_quads =
          ((((0x1) * (p[66]))) + (((0x100) * (((0xFF) & (p[65]))))));
      auto sync_marker =
          ((((0x1) * (p[15]))) + (((0x100) * (p[14]))) +
           (((0x10000) * (p[13]))) + (((0x1000000) * (((0xFF) & (p[12]))))));
      auto space_packet_count =
          ((((0x1) * (p[32]))) + (((0x100) * (p[31]))) +
           (((0x10000) * (p[30]))) + (((0x1000000) * (((0xFF) & (p[29]))))));
      auto pri_count =
          ((((0x1) * (p[36]))) + (((0x100) * (p[35]))) +
           (((0x10000) * (p[34]))) + (((0x1000000) * (((0xFF) & (p[33]))))));
      auto data_delay = ((40) + (((((0x1) * (p[55]))) + (((0x100) * (p[54]))) +
                                  (((0x10000) * (((0xFF) & (p[53]))))))));
      assert((sync_marker) == (0x352EF853));
      try {
        if ((ele) == (ma_ele)) {
          auto n = init_decode_packet(
              packet_idx, ((sar_image) + (((((data_delay) - (mi_data_delay))) +
                                           (((n0) * (ele_count)))))));
          if (!((n) == (((2) * (number_of_quads))))) {
            std::setprecision(3);
            (std::cout) << (std::setw(10))
                        << (((std::chrono::high_resolution_clock::now()
                                  .time_since_epoch()
                                  .count()) -
                             (state._start_time)))
                        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                        << (__func__) << (" ") << ("unexpected number of quads")
                        << (" ") << (std::setw(8)) << (" n=") << (n)
                        << (std::setw(8)) << (" number_of_quads=")
                        << (number_of_quads) << (std::endl);
          };
          (ele_count)++;
        };
      } catch (std::out_of_range e) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("exception") << (" ")
                    << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                    << (std::endl);
      };
      (packet_idx)++;
    };
  };
  delete[](sar_image);
  destroy_mmap();
};