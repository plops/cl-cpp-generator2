
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
      "s1a-ew-raw-s-hv-20191130t152915-20191130t153018-030142-0371ab.dat";
  init_mmap(state._filename);
  init_collect_packet_headers();
  std::array<std::complex<float>, 65535> output;
  auto n = init_decode_packet(0, output);
  destroy_mmap();
};