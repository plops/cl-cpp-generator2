
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <chrono>
#include <iostream>
State state = {};
int main() {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  state._filename =
      "/home/martin/Downloads/"
      "S1A_IW_RAW__0SDV_20191125T135230_20191125T135303_030068_036F1E_6704."
      "SAFE/s1a-iw-raw-s-vv-20191125t135230-20191125t135303-030068-036f1e.dat";
  init_mmap(state._filename);
  init_collect_packet_headers();
  init_process_packet_headers();
  init_decode_packet(0);
  destroy_mmap();
};