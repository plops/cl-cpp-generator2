
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
void init_process_packet_headers() {
  for (auto &e : state._header_data) {
    auto p = e.data();
    auto coarse_time =
        ((((1) * (p[9]))) + (((256) * (p[8]))) + (((65536) * (p[7]))) +
         (((16777216) * (((0xFF) & (p[6]))))));
    auto fine_time = ((((1) * (p[11]))) + (((256) * (((0xFF) & (p[10]))))));
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("header") << (" ")
                << (" coarse_time=") << (coarse_time) << (" fine_time=")
                << (fine_time) << (std::endl);
  };
};