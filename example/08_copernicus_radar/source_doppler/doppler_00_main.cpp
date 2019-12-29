
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
int main() {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  state._filename = "/home/martin//stage/cl-cpp-generator2/example/"
                    "08_copernicus_radar/source/o_range24890_echoes48141.cf";
  init_mmap(state._filename);
  return 0;
};