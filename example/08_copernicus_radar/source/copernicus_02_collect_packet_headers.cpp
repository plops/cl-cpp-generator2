
#include "utils.h"

#include "globals.h"

#include "proto2.h"

extern State state;
#include <array>
#include <cstring>
#include <iostream>
#include <vector>
void destroy_collect_packet_headers() {}
void init_collect_packet_headers() {
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("collect") << (" ") << (std::setw(8))
              << (" state._mmap_data=") << (state._mmap_data) << (std::endl);
  size_t offset = 0;
  while ((offset) < (state._mmap_filesize)) {
    auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
    auto data_length = ((((0x1) * (p[5]))) + (((0x100) * (((0xFF) & (p[4]))))));
    std::array<uint8_t, 62 + 6> data_chunk;
    memcpy(data_chunk.data(), p, ((62) + (6)));
    state._header_offset.push_back(offset);
    state._header_data.push_back(data_chunk);
    (offset) += (((6) + (1) + (data_length)));
  }
}