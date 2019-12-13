
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
State state = {};
int main() {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  state._filename =
      "/home/martin/Downloads/"
      "s1a-ew-raw-s-hh-20191212t201350-20191212t201407-030320-0377ca.dat";
  init_mmap(state._filename);
  init_collect_packet_headers();
  auto packet_idx = 0;
  std::unordered_map<int, int> map_ele;
  std::unordered_map<int, int> map_cal;
  auto cal_count = 0;
  init_sub_commutated_data_decoder();
  remove("./o_anxillary.csv");
  for (auto &e : state._header_data) {
    auto offset = state._header_offset[packet_idx];
    auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
    auto cal_p = ((0x1) & ((p[59]) >> (7)));
    auto ele = ((0xF) & ((p[60]) >> (4)));
    auto cal_type = ((ele) & (7));
    auto number_of_quads =
        ((((0x1) * (p[66]))) + (((0x100) * (((0xFF) & (p[65]))))));
    auto baq_mode = ((0x1F) & ((p[37]) >> (0)));
    auto test_mode = ((0x7) & ((p[21]) >> (4)));
    auto space_packet_count =
        ((((0x1) * (p[32]))) + (((0x100) * (p[31]))) + (((0x10000) * (p[30]))) +
         (((0x1000000) * (((0xFF) & (p[29]))))));
    auto sub_index = ((0xFF) & ((p[26]) >> (0)));
    auto sub_data = ((((0x1) * (p[28]))) + (((0x100) * (((0xFF) & (p[27]))))));
    feed_sub_commutated_data_decoder(sub_data, sub_index, space_packet_count);
    if (cal_p) {
      (cal_count)++;
      (map_cal[((ele) & (7))])++;
      std::setprecision(3);
      (std::cout) << (std::setw(10))
                  << (((std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count()) -
                       (state._start_time)))
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" ") << ("cal") << (" ") << (std::setw(8))
                  << (" cal_p=") << (cal_p) << (std::setw(8)) << (" cal_type=")
                  << (cal_type) << (std::setw(8)) << (" number_of_quads=")
                  << (number_of_quads) << (std::setw(8)) << (" baq_mode=")
                  << (baq_mode) << (std::setw(8)) << (" test_mode=")
                  << (test_mode) << (std::endl);
    } else {
      (map_ele[ele]) += (number_of_quads);
    }
    (packet_idx)++;
  };
  for (auto &cal : map_cal) {
    auto number_of_cal = cal.second;
    auto cal_type = cal.first;
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("map_ele") << (" ") << (std::setw(8))
                << (" cal_type=") << (cal_type) << (std::setw(8))
                << (" number_of_cal=") << (number_of_cal) << (std::endl);
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
              << (" ma=") << (ma) << (std::setw(8)) << (" cal_count=")
              << (cal_count) << (std::endl);
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
      auto cal_p = ((0x1) & ((p[59]) >> (7)));
      auto data_delay = ((40) + (((((0x1) * (p[55]))) + (((0x100) * (p[54]))) +
                                  (((0x10000) * (((0xFF) & (p[53]))))))));
      if (!(cal_p)) {
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
  remove("./o_range.csv");
  remove("./o_cal_range.csv");
  auto cal_n0 = 6000;
  auto cal_iter = 0;
  auto cal_image = new std::complex<float>[((cal_n0) * (cal_count))];
  {
    auto packet_idx = 0;
    auto ele_count = 0;
    for (auto &e : state._header_data) {
      auto offset = state._header_offset[packet_idx];
      auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
      auto azi = ((((0x1) * (p[61]))) + (((0x100) * (((0x3) & (p[60]))))));
      auto baq_n = ((0xFF) & ((p[38]) >> (0)));
      auto baqmod = ((0x1F) & ((p[37]) >> (0)));
      auto cal_mode = ((0x3) & ((p[62]) >> (6)));
      auto cal_p = ((0x1) & ((p[59]) >> (7)));
      auto ecc = ((0xFF) & ((p[20]) >> (0)));
      auto ele = ((0xF) & ((p[60]) >> (4)));
      auto cal_type = ((ele) & (7));
      auto err = ((0x1) & ((p[37]) >> (7)));
      auto number_of_quads =
          ((((0x1) * (p[66]))) + (((0x100) * (((0xFF) & (p[65]))))));
      auto pol = ((0x7) & ((p[59]) >> (4)));
      auto pri_count =
          ((((0x1) * (p[36]))) + (((0x100) * (p[35]))) +
           (((0x10000) * (p[34]))) + (((0x1000000) * (((0xFF) & (p[33]))))));
      auto rank = ((0x1F) & ((p[49]) >> (0)));
      auto rx = ((0xF) & ((p[21]) >> (0)));
      auto rgdec = ((0xFF) & ((p[40]) >> (0)));
      auto signal_type = ((0xF) & ((p[63]) >> (4)));
      auto space_packet_count =
          ((((0x1) * (p[32]))) + (((0x100) * (p[31]))) +
           (((0x10000) * (p[30]))) + (((0x1000000) * (((0xFF) & (p[29]))))));
      auto swath = ((0xFF) & ((p[64]) >> (0)));
      auto swl = ((((0x1) * (p[58]))) + (((0x100) * (p[57]))) +
                  (((0x10000) * (((0xFF) & (p[56]))))));
      auto swst = ((((0x1) * (p[55]))) + (((0x100) * (p[54]))) +
                   (((0x10000) * (((0xFF) & (p[53]))))));
      auto sync_marker =
          ((((0x1) * (p[15]))) + (((0x100) * (p[14]))) +
           (((0x10000) * (p[13]))) + (((0x1000000) * (((0xFF) & (p[12]))))));
      auto tstmod = ((0x7) & ((p[21]) >> (4)));
      auto data_delay = ((40) + (((((0x1) * (p[55]))) + (((0x100) * (p[54]))) +
                                  (((0x10000) * (((0xFF) & (p[53]))))))));
      auto txprr_p = ((0x1) & ((p[42]) >> (7)));
      auto txprr_m = ((((0x1) * (p[43]))) + (((0x100) * (((0x7F) & (p[42]))))));
      auto txpsf_p = ((0x1) & ((p[44]) >> (7)));
      auto txpsf_m = ((((0x1) * (p[45]))) + (((0x100) * (((0x7F) & (p[44]))))));
      auto txpl_ = ((((0x1) * (p[48]))) + (((0x100) * (p[47]))) +
                    (((0x10000) * (((0xFF) & (p[46]))))));
      auto fref = (3.7534723e+1f);
      auto txprr_ = ((pow(-1, txprr_p)) * (txprr_m));
      auto txprr = ((((((fref) * (fref))) / (2097152))) *
                    (pow((-1.e+0f), txprr_p)) * (txprr_m));
      auto txpsf =
          ((((txprr) / (((fref) * (4))))) +
           (((((fref) / (16384))) * (pow((-1.e+0f), txpsf_p)) * (txpsf_m))));
      auto txpl = ((static_cast<double>(txpl_)) / (fref));
      assert((sync_marker) == (0x352EF853));
      try {
        if (cal_p) {
          init_decode_packet_type_a_or_b(
              packet_idx, ((cal_image) + (((cal_n0) * (cal_iter)))));
          {
            std::ofstream outfile;
            outfile.open("./o_cal_range.csv",
                         ((std::ios_base::out) | (std::ios_base::app)));
            if ((0) == (outfile.tellp())) {
              (outfile)
                  << ("azi,baq_n,baqmod,cal_iter,cal_mode,cal_p,cal_type,data_"
                      "delay,number_of_quads,offset,packet_idx,pol,pri_count,"
                      "rank,rgdec,rx,signal_type,space_packet_count,swath,swl,"
                      "swst,tstmod,txpl,txpl_,txprr,txprr_,txpsf")
                  << (std::endl);
            };
            (outfile) << (azi) << (",") << (baq_n) << (",") << (baqmod) << (",")
                      << (cal_iter) << (",") << (cal_mode) << (",") << (cal_p)
                      << (",") << (cal_type) << (",") << (data_delay) << (",")
                      << (number_of_quads) << (",") << (offset) << (",")
                      << (packet_idx) << (",") << (pol) << (",") << (pri_count)
                      << (",") << (rank) << (",") << (rgdec) << (",") << (rx)
                      << (",") << (signal_type) << (",") << (space_packet_count)
                      << (",") << (swath) << (",") << (swl) << (",") << (swst)
                      << (",") << (tstmod) << (",") << (txpl) << (",")
                      << (txpl_) << (",") << (txprr) << (",") << (txprr_)
                      << (",") << (txpsf) << (std::endl);
            outfile.close();
          };
          (cal_iter)++;
        } else {
          if ((ele) == (ma_ele)) {
            auto n = init_decode_packet(
                packet_idx,
                ((sar_image) + (((((data_delay) - (mi_data_delay))) +
                                 (((n0) * (ele_count)))))));
            if (!((n) == (((2) * (number_of_quads))))) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("unexpected number of quads") << (" ")
                          << (std::setw(8)) << (" n=") << (n) << (std::setw(8))
                          << (" number_of_quads=") << (number_of_quads)
                          << (std::endl);
            };
            {
              std::ofstream outfile;
              outfile.open("./o_range.csv",
                           ((std::ios_base::out) | (std::ios_base::app)));
              if ((0) == (outfile.tellp())) {
                (outfile) << ("azi,baq_n,baqmod,cal_iter,cal_mode,cal_p,data_"
                              "delay,ele,ele_count,number_of_quads,offset,"
                              "packet_idx,pol,pri_count,rank,rx,rgdec,signal_"
                              "type,space_packet_count,swath,swl,swst,tstmod,"
                              "txpl,txpl_,txprr,txprr_,txpsf")
                          << (std::endl);
              };
              (outfile) << (azi) << (",") << (baq_n) << (",") << (baqmod)
                        << (",") << (cal_iter) << (",") << (cal_mode) << (",")
                        << (cal_p) << (",") << (data_delay) << (",") << (ele)
                        << (",") << (ele_count) << (",") << (number_of_quads)
                        << (",") << (offset) << (",") << (packet_idx) << (",")
                        << (pol) << (",") << (pri_count) << (",") << (rank)
                        << (",") << (rx) << (",") << (rgdec) << (",")
                        << (signal_type) << (",") << (space_packet_count)
                        << (",") << (swath) << (",") << (swl) << (",") << (swst)
                        << (",") << (tstmod) << (",") << (txpl) << (",")
                        << (txpl_) << (",") << (txprr) << (",") << (txprr_)
                        << (",") << (txpsf) << (std::endl);
              outfile.close();
            };
            (ele_count)++;
          };
        }
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
                    << (std::setw(8)) << (" static_cast<int>(cal_p)=")
                    << (static_cast<int>(cal_p)) << (std::endl);
      };
      (packet_idx)++;
    };
    auto fn = ((std::string("./o_range")) + (std::to_string(n0)) +
               (std::string("_echoes")) + (std::to_string(ele_number_echoes)) +
               (std::string(".cf")));
    auto file = std::ofstream(fn, std::ofstream::binary);
    auto nbytes = ((n0) * (ele_number_echoes) * (sizeof(std::complex<float>)));
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("store") << (" ") << (std::setw(8))
                << (" nbytes=") << (nbytes) << (std::endl);
    file.write(reinterpret_cast<const char *>(sar_image), nbytes);
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("store finished") << (" ")
                << (std::endl);
  };
  delete[](sar_image);
  auto fn = ((std::string("./o_cal_range")) + (std::to_string(cal_n0)) +
             (std::string("_echoes")) + (std::to_string(cal_count)) +
             (std::string(".cf")));
  auto file = std::ofstream(fn, std::ofstream::binary);
  auto nbytes = ((cal_n0) * (cal_count) * (sizeof(std::complex<float>)));
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("store cal") << (" ") << (std::setw(8))
              << (" nbytes=") << (nbytes) << (std::endl);
  file.write(reinterpret_cast<const char *>(cal_image), nbytes);
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("store cal finished") << (" ")
              << (std::endl);
  delete[](cal_image);
  destroy_mmap();
};