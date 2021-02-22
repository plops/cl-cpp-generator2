
#include "utils.h"

#include "globals.h"

#include "proto2.h"

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
      "/media/sdb4/sar/sao_paulo/"
      "s1b-s6-raw-s-vv-20200824t214314-20200824t214345-023070-02bce0.dat";
  init_mmap(state._filename);
  init_collect_packet_headers();
  auto packet_idx = 0;
  std::unordered_map<int, int> map_ele;
  std::unordered_map<int, int> map_cal;
  std::unordered_map<int, int> map_sig;
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
    auto signal_type = ((0xF) & ((p[63]) >> (4)));
    feed_sub_commutated_data_decoder(sub_data, sub_index, space_packet_count);
    (map_sig[signal_type])++;
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
  for (auto &sig : map_sig) {
    auto number_of_sig = sig.second;
    auto sig_type = sig.first;
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("map_sig") << (" ") << (std::setw(8))
                << (" sig_type=") << (sig_type) << (std::setw(8))
                << (" number_of_sig=") << (number_of_sig) << (std::endl);
  };
  auto ma = (-1.0f);
  auto ma_ele = -1;
  for (auto &elevation : map_ele) {
    auto number_of_Mquads = ((elevation.second) / ((1.0e+6f)));
    auto elevation_beam_address = elevation.first;
    if ((ma) < (number_of_Mquads)) {
      ma = number_of_Mquads;
      ma_ele = elevation_beam_address;
    }
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
          if ((data_delay) < (mi_data_delay)) {
            mi_data_delay = data_delay;
          }
          if ((ma_data_delay) < (data_delay)) {
            ma_data_delay = data_delay;
          }
          auto v = ((data_delay) + (((2) * (number_of_quads))));
          if ((ma_data_end) < (v)) {
            ma_data_end = v;
          }
          (map_azi[azi]) += (number_of_quads);
        }
      }
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
      auto number_of_Mquads = ((azi.second) / ((1.0e+6f)));
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
  }
  ele_number_echoes = 10;
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("start big allocation") << (" ")
              << (std::setw(8))
              << (" ((ma_data_end)+(((ma_data_delay)-(mi_data_delay))))=")
              << (((ma_data_end) + (((ma_data_delay) - (mi_data_delay)))))
              << (std::setw(8)) << (" ele_number_echoes=")
              << (ele_number_echoes) << (std::endl);
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
              << (std::setw(8)) << (" (((1.00e-6f))*(n0)*(ele_number_echoes))=")
              << ((((1.00e-6f)) * (n0) * (ele_number_echoes))) << (std::endl);
  remove("./o_all.csv");
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
      auto fref = (37.53472f);
      auto txprr_ = ((pow(-1, txprr_p)) * (txprr_m));
      auto txprr = ((((((fref) * (fref))) / (2097152))) *
                    (pow((-1.0f), txprr_p)) * (txprr_m));
      auto txpsf =
          ((((txprr) / (((fref) * (4))))) +
           (((((fref) / (16384))) * (pow((-1.0f), txpsf_p)) * (txpsf_m))));
      auto txpl = ((static_cast<double>(txpl_)) / (fref));
      assert((sync_marker) == (0x352EF853));
      {
        auto packet_version_number = ((0x7) & ((p[0]) >> (5)));
        auto packet_type = ((0x1) & ((p[0]) >> (4)));
        auto secondary_header_flag = ((0x1) & ((p[0]) >> (3)));
        auto application_process_id_process_id =
            (((((0xF0) & (p[1]))) >> (4)) + (((0x10) * (((0x7) & (p[0]))))));
        auto application_process_id_packet_category = ((0xF) & ((p[1]) >> (0)));
        auto sequence_flags = ((0x3) & ((p[2]) >> (6)));
        auto sequence_count =
            ((((0x1) * (p[3]))) + (((0x100) * (((0x3F) & (p[2]))))));
        auto data_length =
            ((((0x1) * (p[5]))) + (((0x100) * (((0xFF) & (p[4]))))));
        auto coarse_time =
            ((((0x1) * (p[9]))) + (((0x100) * (p[8]))) +
             (((0x10000) * (p[7]))) + (((0x1000000) * (((0xFF) & (p[6]))))));
        auto fine_time =
            ((((0x1) * (p[11]))) + (((0x100) * (((0xFF) & (p[10]))))));
        auto sync_marker =
            ((((0x1) * (p[15]))) + (((0x100) * (p[14]))) +
             (((0x10000) * (p[13]))) + (((0x1000000) * (((0xFF) & (p[12]))))));
        auto data_take_id =
            ((((0x1) * (p[19]))) + (((0x100) * (p[18]))) +
             (((0x10000) * (p[17]))) + (((0x1000000) * (((0xFF) & (p[16]))))));
        auto ecc_number = ((0xFF) & ((p[20]) >> (0)));
        auto ignore_0 = ((0x1) & ((p[21]) >> (7)));
        auto test_mode = ((0x7) & ((p[21]) >> (4)));
        auto rx_channel_id = ((0xF) & ((p[21]) >> (0)));
        auto instrument_configuration_id =
            ((((0x1) * (p[25]))) + (((0x100) * (p[24]))) +
             (((0x10000) * (p[23]))) + (((0x1000000) * (((0xFF) & (p[22]))))));
        auto sub_commutated_index = ((0xFF) & ((p[26]) >> (0)));
        auto sub_commutated_data =
            ((((0x1) * (p[28]))) + (((0x100) * (((0xFF) & (p[27]))))));
        auto space_packet_count =
            ((((0x1) * (p[32]))) + (((0x100) * (p[31]))) +
             (((0x10000) * (p[30]))) + (((0x1000000) * (((0xFF) & (p[29]))))));
        auto pri_count =
            ((((0x1) * (p[36]))) + (((0x100) * (p[35]))) +
             (((0x10000) * (p[34]))) + (((0x1000000) * (((0xFF) & (p[33]))))));
        auto error_flag = ((0x1) & ((p[37]) >> (7)));
        auto ignore_1 = ((0x3) & ((p[37]) >> (5)));
        auto baq_mode = ((0x1F) & ((p[37]) >> (0)));
        auto baq_block_length = ((0xFF) & ((p[38]) >> (0)));
        auto ignore_2 = ((0xFF) & ((p[39]) >> (0)));
        auto range_decimation = ((0xFF) & ((p[40]) >> (0)));
        auto rx_gain = ((0xFF) & ((p[41]) >> (0)));
        auto tx_ramp_rate_polarity = ((0x1) & ((p[42]) >> (7)));
        auto tx_ramp_rate_magnitude =
            ((((0x1) * (p[43]))) + (((0x100) * (((0x7F) & (p[42]))))));
        auto tx_pulse_start_frequency_polarity = ((0x1) & ((p[44]) >> (7)));
        auto tx_pulse_start_frequency_magnitude =
            ((((0x1) * (p[45]))) + (((0x100) * (((0x7F) & (p[44]))))));
        auto tx_pulse_length = ((((0x1) * (p[48]))) + (((0x100) * (p[47]))) +
                                (((0x10000) * (((0xFF) & (p[46]))))));
        auto ignore_3 = ((0x7) & ((p[49]) >> (5)));
        auto rank = ((0x1F) & ((p[49]) >> (0)));
        auto pulse_repetition_interval =
            ((((0x1) * (p[52]))) + (((0x100) * (p[51]))) +
             (((0x10000) * (((0xFF) & (p[50]))))));
        auto sampling_window_start_time =
            ((((0x1) * (p[55]))) + (((0x100) * (p[54]))) +
             (((0x10000) * (((0xFF) & (p[53]))))));
        auto sampling_window_length =
            ((((0x1) * (p[58]))) + (((0x100) * (p[57]))) +
             (((0x10000) * (((0xFF) & (p[56]))))));
        auto sab_ssb_calibration_p = ((0x1) & ((p[59]) >> (7)));
        auto sab_ssb_polarisation = ((0x7) & ((p[59]) >> (4)));
        auto sab_ssb_temp_comp = ((0x3) & ((p[59]) >> (2)));
        auto sab_ssb_ignore_0 = ((0x3) & ((p[59]) >> (0)));
        auto sab_ssb_elevation_beam_address = ((0xF) & ((p[60]) >> (4)));
        auto sab_ssb_ignore_1 = ((0x3) & ((p[60]) >> (2)));
        auto sab_ssb_azimuth_beam_address =
            ((((0x1) * (p[61]))) + (((0x100) * (((0x3) & (p[60]))))));
        auto ses_ssb_cal_mode = ((0x3) & ((p[62]) >> (6)));
        auto ses_ssb_ignore_0 = ((0x1) & ((p[62]) >> (5)));
        auto ses_ssb_tx_pulse_number = ((0x1F) & ((p[62]) >> (0)));
        auto ses_ssb_signal_type = ((0xF) & ((p[63]) >> (4)));
        auto ses_ssb_ignore_1 = ((0x7) & ((p[63]) >> (1)));
        auto ses_ssb_swap = ((0x1) & ((p[63]) >> (0)));
        auto ses_ssb_swath_number = ((0xFF) & ((p[64]) >> (0)));
        auto number_of_quads =
            ((((0x1) * (p[66]))) + (((0x100) * (((0xFF) & (p[65]))))));
        auto ignore_4 = ((0xFF) & ((p[67]) >> (0)));
        {
          std::ofstream outfile;
          outfile.open("./o_all.csv",
                       ((std::ios_base::out) | (std::ios_base::app)));
          if ((0) == (outfile.tellp())) {
            (outfile)
                << ("packet_version_number,packet_type,secondary_header_flag,"
                    "application_process_id_process_id,application_process_id_"
                    "packet_category,sequence_flags,sequence_count,data_length,"
                    "coarse_time,fine_time,sync_marker,data_take_id,ecc_number,"
                    "ignore_0,test_mode,rx_channel_id,instrument_configuration_"
                    "id,sub_commutated_index,sub_commutated_data,space_packet_"
                    "count,pri_count,error_flag,ignore_1,baq_mode,baq_block_"
                    "length,ignore_2,range_decimation,rx_gain,tx_ramp_rate_"
                    "polarity,tx_ramp_rate_magnitude,tx_pulse_start_frequency_"
                    "polarity,tx_pulse_start_frequency_magnitude,tx_pulse_"
                    "length,ignore_3,rank,pulse_repetition_interval,sampling_"
                    "window_start_time,sampling_window_length,sab_ssb_"
                    "calibration_p,sab_ssb_polarisation,sab_ssb_temp_comp,sab_"
                    "ssb_ignore_0,sab_ssb_elevation_beam_address,sab_ssb_"
                    "ignore_1,sab_ssb_azimuth_beam_address,ses_ssb_cal_mode,"
                    "ses_ssb_ignore_0,ses_ssb_tx_pulse_number,ses_ssb_signal_"
                    "type,ses_ssb_ignore_1,ses_ssb_swap,ses_ssb_swath_number,"
                    "number_of_quads,ignore_4,azi,baq_n,baqmod,cal_iter,ele_"
                    "count,cal_mode,cal_p,cal_type,data_delay,offset,packet_"
                    "idx,pol,rgdec,rx,signal_type,swath,swl,swst,tstmod,txpl,"
                    "txpl_,txprr,txprr_,txpsf")
                << (std::endl);
          }
          (outfile) << (packet_version_number) << (",") << (packet_type)
                    << (",") << (secondary_header_flag) << (",")
                    << (application_process_id_process_id) << (",")
                    << (application_process_id_packet_category) << (",")
                    << (sequence_flags) << (",") << (sequence_count) << (",")
                    << (data_length) << (",") << (coarse_time) << (",")
                    << (fine_time) << (",") << (sync_marker) << (",")
                    << (data_take_id) << (",") << (ecc_number) << (",")
                    << (ignore_0) << (",") << (test_mode) << (",")
                    << (rx_channel_id) << (",") << (instrument_configuration_id)
                    << (",") << (sub_commutated_index) << (",")
                    << (sub_commutated_data) << (",") << (space_packet_count)
                    << (",") << (pri_count) << (",") << (error_flag) << (",")
                    << (ignore_1) << (",") << (baq_mode) << (",")
                    << (baq_block_length) << (",") << (ignore_2) << (",")
                    << (range_decimation) << (",") << (rx_gain) << (",")
                    << (tx_ramp_rate_polarity) << (",")
                    << (tx_ramp_rate_magnitude) << (",")
                    << (tx_pulse_start_frequency_polarity) << (",")
                    << (tx_pulse_start_frequency_magnitude) << (",")
                    << (tx_pulse_length) << (",") << (ignore_3) << (",")
                    << (rank) << (",") << (pulse_repetition_interval) << (",")
                    << (sampling_window_start_time) << (",")
                    << (sampling_window_length) << (",")
                    << (sab_ssb_calibration_p) << (",")
                    << (sab_ssb_polarisation) << (",") << (sab_ssb_temp_comp)
                    << (",") << (sab_ssb_ignore_0) << (",")
                    << (sab_ssb_elevation_beam_address) << (",")
                    << (sab_ssb_ignore_1) << (",")
                    << (sab_ssb_azimuth_beam_address) << (",")
                    << (ses_ssb_cal_mode) << (",") << (ses_ssb_ignore_0)
                    << (",") << (ses_ssb_tx_pulse_number) << (",")
                    << (ses_ssb_signal_type) << (",") << (ses_ssb_ignore_1)
                    << (",") << (ses_ssb_swap) << (",")
                    << (ses_ssb_swath_number) << (",") << (number_of_quads)
                    << (",") << (ignore_4) << (",") << (azi) << (",") << (baq_n)
                    << (",") << (baqmod) << (",") << (cal_iter) << (",")
                    << (ele_count) << (",") << (cal_mode) << (",") << (cal_p)
                    << (",") << (cal_type) << (",") << (data_delay) << (",")
                    << (offset) << (",") << (packet_idx) << (",") << (pol)
                    << (",") << (rgdec) << (",") << (rx) << (",")
                    << (signal_type) << (",") << (swath) << (",") << (swl)
                    << (",") << (swst) << (",") << (tstmod) << (",") << (txpl)
                    << (",") << (txpl_) << (",") << (txprr) << (",") << (txprr_)
                    << (",") << (txpsf) << (std::endl);
          outfile.close();
        }
      }
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
                  << ("azi,baq_n,baqmod,cal_iter,ele_count,cal_mode,cal_p,cal_"
                      "type,data_delay,number_of_quads,offset,packet_idx,pol,"
                      "pri_count,rank,rgdec,rx,signal_type,space_packet_count,"
                      "swath,swl,swst,tstmod,txpl,txpl_,txprr,txprr_,txpsf")
                  << (std::endl);
            }
            (outfile) << (azi) << (",") << (baq_n) << (",") << (baqmod) << (",")
                      << (cal_iter) << (",") << (ele_count) << (",")
                      << (cal_mode) << (",") << (cal_p) << (",") << (cal_type)
                      << (",") << (data_delay) << (",") << (number_of_quads)
                      << (",") << (offset) << (",") << (packet_idx) << (",")
                      << (pol) << (",") << (pri_count) << (",") << (rank)
                      << (",") << (rgdec) << (",") << (rx) << (",")
                      << (signal_type) << (",") << (space_packet_count) << (",")
                      << (swath) << (",") << (swl) << (",") << (swst) << (",")
                      << (tstmod) << (",") << (txpl) << (",") << (txpl_)
                      << (",") << (txprr) << (",") << (txprr_) << (",")
                      << (txpsf) << (std::endl);
            outfile.close();
          }
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
            }
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
              }
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
            }
            (ele_count)++;
          }
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
    auto fn = ((std::string("/media/sdb4/sar/o_range")) + (std::to_string(n0)) +
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
                << (__func__) << (" ") << ("store echo") << (" ")
                << (std::setw(8)) << (" nbytes=") << (nbytes) << (std::endl);
    file.write(reinterpret_cast<const char *>(sar_image), nbytes);
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("store echo finished") << (" ")
                << (std::endl);
  }
  delete[](sar_image);
  auto fn = ((std::string("/media/sdb4/sar/o_cal_range")) +
             (std::to_string(cal_n0)) + (std::string("_echoes")) +
             (std::to_string(cal_count)) + (std::string(".cf")));
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
}