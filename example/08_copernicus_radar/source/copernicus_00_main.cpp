
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
      "S1A_IW_RAW__0SDV_20181106T135244_20181106T135316_024468_02AEB9_3552."
      "SAFE/s1a-iw-raw-s-vv-20181106t135244-20181106t135316-024468-02aeb9.dat";
  init_mmap(state._filename);
  init_collect_packet_headers();
  init_process_packet_headers();
  init_decode_packet(0);
  destroy_mmap();
};