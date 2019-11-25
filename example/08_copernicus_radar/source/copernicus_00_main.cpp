
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
  init_mmap(
      "/home/martin/Downloads/"
      "S1A_IW_RAW__0SDV_20191030T055015_20191030T055047_029684_0361B3_78C6."
      "SAFE/s1a-iw-raw-s-vv-20191030t055015-20191030t055047-029684-0361b3.dat");
  init_collect_packet_headers();
  init_process_packet_headers();
  destroy_mmap();
};