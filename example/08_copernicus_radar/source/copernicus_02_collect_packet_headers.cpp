
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <array>
#include <cstring>
#include <iostream>
#include <vector>
void destroy_collect_packet_headers() {}
void init_collect_packet_headers() {
  (std::cout) << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("collect") << (" ")
              << (" state._mmap_data=") << (state._mmap_data) << (std::endl);
  size_t offset = 0;
  while (offset < state._mmap_filesize) {
    auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
    auto data_length = ((((1) * (p[5]))) + (((256) * (((0xFF) & (p[4]))))));
    auto sync_marker =
        ((((1) * (p[15]))) + (((256) * (p[14]))) + (((65536) * (p[13]))) +
         (((16777216) * (((0xFF) & (p[12]))))));
    std::array<uint8_t, 62 + 6> data_chunk;
    memcpy(data_chunk.data(), p, ((62) + (6)));
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("len") << (" ") << (" offset=")
                << (offset) << (" data_length=") << (data_length)
                << (" sync_marker=") << (sync_marker) << (std::endl);
    state._header_offset.push_back(offset);
    state._header_data.push_back(data_chunk);
    (offset) += (((6) + (1) + (data_length)));
  };
};