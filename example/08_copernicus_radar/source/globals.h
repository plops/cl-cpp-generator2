#ifndef GLOBALS_H

#define GLOBALS_H

#include <chrono>
struct State {
  typeof(std::chrono::high_resolution_clock::now().time_since_epoch().count())
      _start_time;
  void *_header_data;
  size_t _mmap_filesize;
  void *_mmap_data;
};
typedef struct State State;

#endif
