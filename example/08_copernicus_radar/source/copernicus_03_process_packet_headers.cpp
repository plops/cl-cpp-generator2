
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <unistd.h>
void init_process_packet_headers() {
  auto p0 = state._header_data[0].data();
  auto coarse_time0 =
      ((((1) * (p0[9]))) + (((256) * (p0[8]))) + (((65536) * (p0[7]))) +
       (((16777216) * (((0xFF) & ((p0[6]) >> (0)))))));
  auto fine_time0 =
      (((1.52587890625e-5)) *
       ((((5.e-1f)) +
         (((((1) * (p0[11]))) + (((256) * (((0xFF) & ((p0[10]) >> (0)))))))))));
  auto time0 = ((coarse_time0) + (fine_time0));
  for (auto &e : state._header_data) {
    auto p = e.data();
    auto fref = (3.7534723e+1f);
    auto coarse_time =
        ((((1) * (p[9]))) + (((256) * (p[8]))) + (((65536) * (p[7]))) +
         (((16777216) * (((0xFF) & ((p[6]) >> (0)))))));
    auto fine_time =
        ((((1) * (p[11]))) + (((256) * (((0xFF) & ((p[10]) >> (0)))))));
    auto ftime = (((1.52587890625e-5)) * ((((5.e-1f)) + (fine_time))));
    auto time = ((((coarse_time) + (ftime))) - (time0));
    auto swst = ((((((1) * (p[55]))) + (((256) * (p[54]))) +
                   (((65536) * (((0xFF) & ((p[53]) >> (0)))))))) /
                 (fref));
    auto azi = ((((1) * (p[61]))) + (((256) * (((0x3) & ((p[60]) >> (6)))))));
    auto count =
        ((((1) * (p[32]))) + (((256) * (p[31]))) + (((65536) * (p[30]))) +
         (((16777216) * (((0xFF) & ((p[29]) >> (0)))))));
    auto pri_count =
        ((((1) * (p[36]))) + (((256) * (p[35]))) + (((65536) * (p[34]))) +
         (((16777216) * (((0xFF) & ((p[33]) >> (0)))))));
    auto pri = ((((((1) * (p[52]))) + (((256) * (p[51]))) +
                  (((65536) * (((0xFF) & ((p[50]) >> (0)))))))) /
                (fref));
    auto rank = ((0x1F) & ((p[49]) >> (3)));
    auto rank2 = static_cast<int>(p[((49))]);
    auto baqmod = ((0x1F) & ((p[37]) >> (3)));
    auto baq_n = ((0xFF) & ((p[38]) >> (0)));
    auto sync_marker =
        ((((1) * (p[15]))) + (((256) * (p[14]))) + (((65536) * (p[13]))) +
         (((16777216) * (((0xFF) & ((p[12]) >> (0)))))));
    auto sync2 =
        ((((16777216) * (((255) & (static_cast<int>(p[((12) + (0))])))))) +
         (((65536) * (((255) & (static_cast<int>(p[((12) + (1))])))))) +
         (((256) * (((255) & (static_cast<int>(p[((12) + (2))])))))) +
         (((1) * (((255) & (static_cast<int>(p[((12) + (3))])))))));
    auto baqmod2 = static_cast<int>(p[37]);
    auto err = ((0x1) & ((p[37]) >> (0)));
    auto tstmod = ((0x7) & ((p[21]) >> (1)));
    auto rx = ((0xF) & ((p[21]) >> (4)));
    auto ecc = ((0xFF) & ((p[20]) >> (0)));
    auto pol = ((0x7) & ((p[59]) >> (1)));
    auto signal_type = ((0xF) & ((p[63]) >> (0)));
    auto swath = ((0xFF) & ((p[64]) >> (0)));
    auto ele = ((0xF) & ((p[60]) >> (0)));
    {
      auto v = static_cast<int>(((0x7) & ((p[0]) >> (0))));
      (std::cout) << (std::setw(42)) << ("packet-version-number ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1) & ((p[0]) >> (3))));
      (std::cout) << (std::setw(42)) << ("packet-type ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1) & ((p[0]) >> (4))));
      (std::cout) << (std::setw(42)) << ("secondary-header-flag ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x7F) & ((p[0]) >> (5))));
      (std::cout) << (std::setw(42)) << ("application-process-id-process-id ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xF) & ((p[1]) >> (4))));
      (std::cout) << (std::setw(42))
                  << ("application-process-id-packet-category ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x3) & ((p[2]) >> (0))));
      (std::cout) << (std::setw(42)) << ("sequence-flags ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[3]))) + (((256) * (((0x3F) & ((p[2]) >> (2))))))));
      (std::cout) << (std::setw(42)) << ("sequence-count ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[5]))) + (((256) * (((0xFF) & ((p[4]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("data-length ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[9]))) + (((256) * (p[8]))) + (((65536) * (p[7]))) +
           (((16777216) * (((0xFF) & ((p[6]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("coarse-time ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[11]))) + (((256) * (((0xFF) & ((p[10]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("fine-time ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[15]))) + (((256) * (p[14]))) + (((65536) * (p[13]))) +
           (((16777216) * (((0xFF) & ((p[12]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("sync-marker ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[19]))) + (((256) * (p[18]))) + (((65536) * (p[17]))) +
           (((16777216) * (((0xFF) & ((p[16]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("data-take-id ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xFF) & ((p[20]) >> (0))));
      (std::cout) << (std::setw(42)) << ("ecc-number ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1) & ((p[21]) >> (0))));
      (std::cout) << (std::setw(42)) << ("ignore-0 ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x7) & ((p[21]) >> (1))));
      (std::cout) << (std::setw(42)) << ("test-mode ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xF) & ((p[21]) >> (4))));
      (std::cout) << (std::setw(42)) << ("rx-channel-id ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[25]))) + (((256) * (p[24]))) + (((65536) * (p[23]))) +
           (((16777216) * (((0xFF) & ((p[22]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("instrument-configuration-id ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xFF) & ((p[26]) >> (0))));
      (std::cout) << (std::setw(42)) << ("sub-commutated-index ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[28]))) + (((256) * (((0xFF) & ((p[27]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("sub-commutated-data ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[32]))) + (((256) * (p[31]))) + (((65536) * (p[30]))) +
           (((16777216) * (((0xFF) & ((p[29]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("space-packet-count ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[36]))) + (((256) * (p[35]))) + (((65536) * (p[34]))) +
           (((16777216) * (((0xFF) & ((p[33]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("pri-count ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1) & ((p[37]) >> (0))));
      (std::cout) << (std::setw(42)) << ("error-flag ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x3) & ((p[37]) >> (1))));
      (std::cout) << (std::setw(42)) << ("ignore-1 ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1F) & ((p[37]) >> (3))));
      (std::cout) << (std::setw(42)) << ("baq-mode ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xFF) & ((p[38]) >> (0))));
      (std::cout) << (std::setw(42)) << ("baq-block-length ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xFF) & ((p[39]) >> (0))));
      (std::cout) << (std::setw(42)) << ("ignore-2 ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xFF) & ((p[40]) >> (0))));
      (std::cout) << (std::setw(42)) << ("range-decimation ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xFF) & ((p[41]) >> (0))));
      (std::cout) << (std::setw(42)) << ("rx-gain ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1) & ((p[42]) >> (0))));
      (std::cout) << (std::setw(42)) << ("tx-ramp-rate-polarity ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[43]))) + (((256) * (((0x7F) & ((p[42]) >> (1))))))));
      (std::cout) << (std::setw(42)) << ("tx-ramp-rate-magnitude ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1) & ((p[44]) >> (0))));
      (std::cout) << (std::setw(42)) << ("tx-pulse-start-frequency-polarity ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[45]))) + (((256) * (((0x7F) & ((p[44]) >> (1))))))));
      (std::cout) << (std::setw(42)) << ("tx-pulse-start-frequency-magnitude ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((((1) * (p[48]))) + (((256) * (p[47]))) +
                                 (((65536) * (((0xFF) & ((p[46]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("tx-pulse-length ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x7) & ((p[49]) >> (0))));
      (std::cout) << (std::setw(42)) << ("ignore-3 ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1F) & ((p[49]) >> (3))));
      (std::cout) << (std::setw(42)) << ("rank ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((((1) * (p[52]))) + (((256) * (p[51]))) +
                                 (((65536) * (((0xFF) & ((p[50]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("pulse-repetition-interval ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((((1) * (p[55]))) + (((256) * (p[54]))) +
                                 (((65536) * (((0xFF) & ((p[53]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("sampling-window-start-time ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((((1) * (p[58]))) + (((256) * (p[57]))) +
                                 (((65536) * (((0xFF) & ((p[56]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("sampling-window-length ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1) & ((p[59]) >> (0))));
      (std::cout) << (std::setw(42)) << ("sab-ssb-calibration-p ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x7) & ((p[59]) >> (1))));
      (std::cout) << (std::setw(42)) << ("sab-ssb-polarisation ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x3) & ((p[59]) >> (4))));
      (std::cout) << (std::setw(42)) << ("sab-ssb-temp-comp ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x3) & ((p[59]) >> (6))));
      (std::cout) << (std::setw(42)) << ("sab-ssb-ignore-0 ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xF) & ((p[60]) >> (0))));
      (std::cout) << (std::setw(42)) << ("sab-ssb-elevation-beam-address ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x3) & ((p[60]) >> (4))));
      (std::cout) << (std::setw(42)) << ("sab-ssb-ignore-1 ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[61]))) + (((256) * (((0x3) & ((p[60]) >> (6))))))));
      (std::cout) << (std::setw(42)) << ("sab-ssb-azimuth-beam-address ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x3) & ((p[62]) >> (0))));
      (std::cout) << (std::setw(42)) << ("ses-ssb-cal-mode ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1) & ((p[62]) >> (2))));
      (std::cout) << (std::setw(42)) << ("ses-ssb-ignore-0 ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1F) & ((p[62]) >> (3))));
      (std::cout) << (std::setw(42)) << ("ses-ssb-tx-pulse-number ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xF) & ((p[63]) >> (0))));
      (std::cout) << (std::setw(42)) << ("ses-ssb-signal-type ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x7) & ((p[63]) >> (4))));
      (std::cout) << (std::setw(42)) << ("ses-ssb-ignore-1 ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0x1) & ((p[63]) >> (7))));
      (std::cout) << (std::setw(42)) << ("ses-ssb-swap ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xFF) & ((p[64]) >> (0))));
      (std::cout) << (std::setw(42)) << ("ses-ssb-swath-number ")
                  << (std::setw(12)) << (std::dec) << (v) << (std::setw(12))
                  << (std::hex) << (v) << (std::endl);
    };
    {
      auto v = static_cast<int>(
          ((((1) * (p[66]))) + (((256) * (((0xFF) & ((p[65]) >> (0))))))));
      (std::cout) << (std::setw(42)) << ("number-of-quads ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    {
      auto v = static_cast<int>(((0xFF) & ((p[67]) >> (0))));
      (std::cout) << (std::setw(42)) << ("ignore-4 ") << (std::setw(12))
                  << (std::dec) << (v) << (std::setw(12)) << (std::hex) << (v)
                  << (std::endl);
    };
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("") << (" ") << (std::setw(8))
                << (" time=") << (time) << (std::setw(8)) << (" std::hex=")
                << (std::hex) << (std::setw(8)) << (" err=") << (err)
                << (std::setw(8)) << (" swst=") << (swst) << (std::setw(8))
                << (" coarse_time=") << (coarse_time) << (std::setw(8))
                << (" fine_time=") << (fine_time) << (std::setw(8))
                << (" swath=") << (swath) << (std::setw(8)) << (" count=")
                << (count) << (std::setw(8)) << (" pri_count=") << (pri_count)
                << (std::setw(8)) << (" rank=") << (rank) << (std::setw(8))
                << (" rank2=") << (rank2) << (std::setw(8)) << (" pri=")
                << (pri) << (std::setw(8)) << (" baqmod=") << (baqmod)
                << (std::setw(8)) << (" baq_n=") << (baq_n) << (std::setw(8))
                << (" sync2=") << (sync2) << (std::setw(8)) << (" sync_marker=")
                << (sync_marker) << (std::setw(8)) << (" baqmod2=") << (baqmod2)
                << (std::setw(8)) << (" tstmod=") << (tstmod) << (std::setw(8))
                << (" azi=") << (azi) << (std::setw(8)) << (" ele=") << (ele)
                << (std::setw(8)) << (" rx=") << (rx) << (std::setw(8))
                << (" pol=") << (pol) << (std::setw(8)) << (" ecc=") << (ecc)
                << (std::setw(8)) << (" signal_type=") << (signal_type)
                << (std::endl);
    usleep(32000);
    (std::cout) << ("\033[2J\033[1;1H") << (std::flush);
  };
};