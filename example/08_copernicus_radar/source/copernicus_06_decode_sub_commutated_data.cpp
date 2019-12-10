
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <cstring>

void init_sub_commutated_data_decoder() {
  state._ancillary_data_index = 0;
  for (int i = 0; i < 64; (i) += (1)) {
    state._ancillary_data_valid.at(i) = false;
  }
}
bool feed_sub_commutated_data_decoder(uint16_t word, int idx) {
  state._ancillary_data_index = idx;
  state._ancillary_data.at(state._ancillary_data_index) = word;
  state._ancillary_data.at(state._ancillary_data_index) = true;
  if ((state._ancillary_data_index) == (64)) {
    memcpy(reinterpret_cast<void *>(&(state._ancillary_decoded)),
           reinterpret_cast<void *>(state._ancillary_data.data()),
           sizeof(state._ancillary_data));
    init_sub_commutated_data_decoder();
    return true;
  } else {
    return false;
  }
};