
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
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
        (((1.52587890625e-5)) *
         ((((5.e-1f)) +
           (((((1) * (p[11]))) + (((256) * (((0xFF) & ((p[10]) >> (0)))))))))));
    auto time = ((((coarse_time) + (fine_time))) - (time0));
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
    auto sync_marker =
        ((((1) * (p[15]))) + (((256) * (p[14]))) + (((65536) * (p[13]))) +
         (((16777216) * (((0xFF) & ((p[12]) >> (0)))))));
    auto sync2 =
        ((((16777216) * (((255) & (static_cast<int>(p[((12) + (0))])))))) +
         (((65536) * (((255) & (static_cast<int>(p[((12) + (1))])))))) +
         (((256) * (((255) & (static_cast<int>(p[((12) + (2))])))))) +
         (((1) * (((255) & (static_cast<int>(p[((12) + (3))])))))));
    auto baqmod2 = ((31) & ((p[37]) >> (3)));
    auto tstmod = ((0x7) & ((p[21]) >> (1)));
    auto rx = ((0xF) & ((p[21]) >> (4)));
    auto pol = ((0x7) & ((p[59]) >> (1)));
    auto swath = ((0xFF) & ((p[64]) >> (0)));
    auto ele = ((0xF) & ((p[60]) >> (0)));
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("") << (" ") << (std::setw(8))
                << (" time=") << (time) << (std::setw(8)) << (" swst=")
                << (swst) << (std::setw(8)) << (" swath=") << (swath)
                << (std::setw(8)) << (" count=") << (count) << (std::setw(8))
                << (" pri_count=") << (pri_count) << (std::setw(8))
                << (" rank=") << (rank) << (std::setw(8)) << (" rank2=")
                << (rank2) << (std::setw(8)) << (" pri=") << (pri)
                << (std::setw(8)) << (" baqmod=") << (baqmod) << (std::setw(8))
                << (" sync2=") << (sync2) << (std::setw(8)) << (" sync_marker=")
                << (sync_marker) << (std::setw(8)) << (" baqmod2=") << (baqmod2)
                << (std::setw(8)) << (" tstmod=") << (tstmod) << (std::setw(8))
                << (" azi=") << (azi) << (std::setw(8)) << (" ele=") << (ele)
                << (std::setw(8)) << (" rx=") << (rx) << (std::setw(8))
                << (" pol=") << (pol) << (std::endl);
  };
};