
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;

void init_sequential_bit_function(sequential_bit_t *seq_state,
                                  size_t byte_pos) {
  seq_state->data = &(static_cast<uint8_t *>(state._mmap_data)[byte_pos]);
  seq_state->current_bit_count = 0;
}
inline bool get_sequential_bit(sequential_bit_t *seq_state) {
  auto current_byte = *(seq_state->data);
  auto res = static_cast<bool>(
      (((current_byte) >> (seq_state->current_bit_count)) && (1)));
  if (7 < seq_state->current_bit_count) {
    seq_state->current_bit_count = 0;
    (seq_state->data)++;
  };
  return res;
};
int get_brc(sequential_bit_t *s) {
  // note: evaluation order is crucial
  return ((((4) * (get_sequential_bit(s)))) +
          (((2) * (get_sequential_bit(s)))) +
          (((1) * (get_sequential_bit(s)))));
}
void init_decode_packet(int packet_idx) {
  auto header = state._header_data[packet_idx].data();
  auto offset = state._header_offset[packet_idx];
  auto number_of_quads =
      ((((1) * (header[66]))) + (((256) * (((0xFF) & (header[65]))))));
  auto data = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
  auto decoded_symbols = 0;
  auto number_of_baq_blocks = ((((2) * (number_of_quads))) / (256));
  sequential_bit_t s;
  init_sequential_bit_function(
      &s, ((state._header_offset[packet_idx]) + (62) + (6)));
  for (int block = 0; decoded_symbols < number_of_quads;) {
    auto brc = get_brc(&s);
  };
};