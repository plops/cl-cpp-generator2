#ifndef GLOBALS_H

#define GLOBALS_H

#include <mutex>

#include <chrono>
struct State {
  typeof(std::chrono::high_resolution_clock::now().time_since_epoch().count())
      _start_time;
  std::string _code_generation_time;
  std::string _code_repository;
  std::string _main_version;
};
;

#endif
