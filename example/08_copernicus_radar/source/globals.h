#ifndef GLOBALS_H

#define GLOBALS_H

#include <chrono>
struct State {
  typeof(std::chrono::high_resolution_clock::now().time_since_epoch().count())
      _start_time;
  int _ancillary_data_index;
  ancillary_data_t _ancillary_decoded;
  std::array<bool, 65> _ancillary_data_valid;
  std::array<uint16_t, 65> _ancillary_data;
  std::vector<size_t> _header_offset;
  std::vector<std::array<uint8_t, 62 + 6>> _header_data;
  size_t _mmap_filesize;
  void *_mmap_data;
  char const *_filename;
};
typedef struct State State;

#endif
