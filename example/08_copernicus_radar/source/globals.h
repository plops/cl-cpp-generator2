#ifndef GLOBALS_H

#define GLOBALS_H

#include <chrono>
struct State {
  typeof(std::chrono::high_resolution_clock::now().time_since_epoch().count())
      _start_time;
  std::vector<size_t> _header_offset;
  std::vector<std::array<uint8_t, 62 + 6>> _header_data;
  size_t _mmap_filesize;
  void *_mmap_data;
};
typedef struct State State;

#endif
