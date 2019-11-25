
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <array>
#include <iostream>
struct space_packet_header_info_t {
  std::array<uint8_t, 62 + 6> head;
  size_t offset;
};
typedef struct space_packet_header_info_t space_packet_header_info_t;
void destroy_collect_packet_headers() {}
void init_collect_packet_headers() {
  (std::cout) << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("collect") << (" ")
              << (" state._mmap_data=") << (state._mmap_data) << (std::endl);
  auto offset = 0;
  for (int i = 0; i < 10; (i) += (1)) {
    auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
    auto data_length = ((((1) * (p[5]))) + (((256) * (((0xFF) & (p[4]))))));
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("len") << (" ") << (" offset=")
                << (offset) << (" data_length=") << (data_length)
                << (std::endl);
    (offset) += (((6) + (1) + (data_length)));
  };
};