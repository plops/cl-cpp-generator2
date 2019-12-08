
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <cassert>
int get_data_type_a_or_b(sequential_bit_t *s) {
  return ((((0x200) * (get_sequential_bit(s)))) +
          (((0x100) * (get_sequential_bit(s)))) +
          (((0x80) * (get_sequential_bit(s)))) +
          (((0x40) * (get_sequential_bit(s)))) +
          (((0x20) * (get_sequential_bit(s)))) +
          (((0x10) * (get_sequential_bit(s)))) +
          (((0x8) * (get_sequential_bit(s)))) +
          (((0x4) * (get_sequential_bit(s)))) +
          (((0x2) * (get_sequential_bit(s)))) +
          (((0x1) * (get_sequential_bit(s)))));
};
int init_decode_packet_type_a_or_b(int packet_idx,
                                   std::complex<float> *output) {
  // packet_idx .. index of space packet 0 ..
  // mi_data_delay .. if -1, ignore; otherwise it is assumed to be the smallest
  // delay in samples between tx pulse start and data acquisition and will be
  // used to compute a sample offset in output so that all echos of one sar
  // image are aligned to the same time offset output .. array of complex
  // numbers return value: number of complex data samples written
  auto header = state._header_data[packet_idx].data();
  auto offset = state._header_offset[packet_idx];
  auto number_of_quads =
      ((((0x1) * (header[66]))) + (((0x100) * (((0xFF) & (header[65]))))));
  auto baq_block_length = ((8) * (((1) + (((0xFF) & ((header[38]) >> (0)))))));
  auto number_of_words = static_cast<int>(
      round(ceil((((((1.e+1f)) * (number_of_quads))) / (16)))));
  auto baq_mode = ((0x1F) & ((header[37]) >> (0)));
  auto fref = (3.7534723e+1f);
  auto swst = ((((((0x1) * (header[55]))) + (((0x100) * (header[54]))) +
                 (((0x10000) * (((0xFF) & (header[53]))))))) /
               (fref));
  auto delta_t_suppressed = (((3.2e+2)) / (((8) * (fref))));
  auto data_delay_us = ((swst) + (delta_t_suppressed));
  auto data_delay =
      ((40) + (((((0x1) * (header[55]))) + (((0x100) * (header[54]))) +
                (((0x10000) * (((0xFF) & (header[53]))))))));
  auto data = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
  assert((((0) == (baq_mode))));
  sequential_bit_t s;
  init_sequential_bit_function(
      &s, ((state._header_offset[packet_idx]) + (62) + (6)));
  auto decoded_ie_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_ie_symbols_a;
  for (int i = 0; i < MAX_NUMBER_QUADS; (i) += (1)) {
    decoded_ie_symbols_a[i] = (0.0e+0f);
  };
  // parse ie data
  for (int i = 0; i < number_of_quads; (i) += (1)) {
    auto smcode = get_data_type_a_or_b(&s);
    auto sign_bit = (((smcode) & (0x200))) >> (9);
    auto mcode = ((smcode) & (0x1FF));
    auto scode = ((powf((-1.e+0f), sign_bit)) * (mcode));
    decoded_ie_symbols_a[decoded_ie_symbols] = scode;
    (decoded_ie_symbols)++;
  }
  consume_padding_bits(&s);
  auto decoded_io_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_io_symbols_a;
  for (int i = 0; i < MAX_NUMBER_QUADS; (i) += (1)) {
    decoded_io_symbols_a[i] = (0.0e+0f);
  };
  // parse io data
  for (int i = 0; i < number_of_quads; (i) += (1)) {
    auto smcode = get_data_type_a_or_b(&s);
    auto sign_bit = (((smcode) & (0x200))) >> (9);
    auto mcode = ((smcode) & (0x1FF));
    auto scode = ((powf((-1.e+0f), sign_bit)) * (mcode));
    decoded_io_symbols_a[decoded_io_symbols] = scode;
    (decoded_io_symbols)++;
  }
  consume_padding_bits(&s);
  auto decoded_qe_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_qe_symbols_a;
  for (int i = 0; i < MAX_NUMBER_QUADS; (i) += (1)) {
    decoded_qe_symbols_a[i] = (0.0e+0f);
  };
  // parse qe data
  for (int i = 0; i < number_of_quads; (i) += (1)) {
    auto smcode = get_data_type_a_or_b(&s);
    auto sign_bit = (((smcode) & (0x200))) >> (9);
    auto mcode = ((smcode) & (0x1FF));
    auto scode = ((powf((-1.e+0f), sign_bit)) * (mcode));
    decoded_qe_symbols_a[decoded_qe_symbols] = scode;
    (decoded_qe_symbols)++;
  }
  consume_padding_bits(&s);
  auto decoded_qo_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_qo_symbols_a;
  for (int i = 0; i < MAX_NUMBER_QUADS; (i) += (1)) {
    decoded_qo_symbols_a[i] = (0.0e+0f);
  };
  // parse qo data
  for (int i = 0; i < number_of_quads; (i) += (1)) {
    auto smcode = get_data_type_a_or_b(&s);
    auto sign_bit = (((smcode) & (0x200))) >> (9);
    auto mcode = ((smcode) & (0x1FF));
    auto scode = ((powf((-1.e+0f), sign_bit)) * (mcode));
    decoded_qo_symbols_a[decoded_qo_symbols] = scode;
    (decoded_qo_symbols)++;
  }
  consume_padding_bits(&s);
  for (int i = 0; i < decoded_ie_symbols; (i) += (1)) {
    output[((2) * (i))].real(decoded_ie_symbols_a[i]);
    output[((2) * (i))].imag(decoded_qe_symbols_a[i]);
    output[((1) + (((2) * (i))))].real(decoded_io_symbols_a[i]);
    output[((1) + (((2) * (i))))].imag(decoded_qo_symbols_a[i]);
  }
  auto n = ((decoded_ie_symbols) + (decoded_io_symbols));
  return n;
};