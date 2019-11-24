#ifndef GLOBALS_H

#define GLOBALS_H

#include <chrono>
struct State {
  typeof(std::chrono::high_resolution_clock::now().time_since_epoch().count())
      _start_time;
};
typedef struct State State;

#endif
