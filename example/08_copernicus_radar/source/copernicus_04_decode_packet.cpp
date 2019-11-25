
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
void init_decode_packet(int packet_idx) {
  auto header = state._header_data[packet_idx].data();
  auto offset = state._header_offset[packet_idx];
  auto number_of_quads =
      ((((1) * (header[66]))) + (((256) * (((0xFF) & (header[65]))))));
};