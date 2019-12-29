
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
State state = {};
double now() {
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return (((double)tp.tv_sec) + ((((1.e-9)) * (tp.tv_nsec))));
}
void mainLoop() {
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("mainLoop") << (" ") << (std::endl);
  while (!(glfwWindowShouldClose(state._window))) {
    glfwPollEvents();
  }
}
void run() {
  initWindow();
  mainLoop();
};
int main() {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  state._filename = "/home/martin//stage/cl-cpp-generator2/example/"
                    "08_copernicus_radar/source/o_range24890_echoes48141.cf";
  init_mmap(state._filename);
  run();
  cleanupWindow();
  return 0;
};