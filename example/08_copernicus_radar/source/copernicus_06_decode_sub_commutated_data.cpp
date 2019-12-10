
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;

void init_sub_commutated_data_decoder() { state._ancillary_data_index = 0; }
bool feed_sub_commutated_data_decoder(uint16_t word) {
  state._ancillary_data.at(state._ancillary_data_index) = word;
  (state._ancillary_data_index)++;
  if ((state._ancillary_data_index) == (64)) {
    state._ancillary_data_index = 0;
    return true;
  } else {
    return false;
  }
};