
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <cassert>

void init_sequential_bit_function(sequential_bit_t *seq_state,
                                  size_t byte_pos) {
  seq_state->data = &(static_cast<uint8_t *>(state._mmap_data)[byte_pos]);
  seq_state->current_bit_count = 0;
}
inline bool get_sequential_bit(sequential_bit_t *seq_state) {
  auto current_byte = *(seq_state->data);
  auto res = static_cast<bool>(
      (((current_byte) >> (((7) - (seq_state->current_bit_count)))) & (1)));
  (seq_state->current_bit_count)++;
  if (7 < seq_state->current_bit_count) {
    seq_state->current_bit_count = 0;
    (seq_state->data)++;
  };
  return res;
};
inline int get_threshold_index(sequential_bit_t *s) {
  return ((((0x80) * (get_sequential_bit(s)))) +
          (((0x40) * (get_sequential_bit(s)))) +
          (((0x20) * (get_sequential_bit(s)))) +
          (((0x10) * (get_sequential_bit(s)))) +
          (((0x8) * (get_sequential_bit(s)))) +
          (((0x4) * (get_sequential_bit(s)))) +
          (((0x2) * (get_sequential_bit(s)))) +
          (((0x1) * (get_sequential_bit(s)))));
}
inline int get_bit_rate_code(sequential_bit_t *s) {
  // note: evaluation order is crucial
  return ((((0x4) * (get_sequential_bit(s)))) +
          (((0x2) * (get_sequential_bit(s)))) +
          (((0x1) * (get_sequential_bit(s)))));
}
inline void consume_padding_bits(sequential_bit_t *s) {
  auto byte_offset = static_cast<int>(
      ((s->data) - (static_cast<uint8_t *>(state._mmap_data))));
  // make sure we are at first bit of an even byte in the next read
  s->current_bit_count = 0;
  if ((0) == (byte_offset % 2)) {
    // we are in an even byte
    (s->data) += (2);
  } else {
    // we are in an odd byte
    (s->data) += (1);
  };
}
inline int decode_huffman_brc0(sequential_bit_t *s) {
  if (get_sequential_bit(s)) {
    if (get_sequential_bit(s)) {
      if (get_sequential_bit(s)) {
        return 3;
      } else {
        return 2;
      }
    } else {
      return 1;
    }
  } else {
    return 0;
  }
}
inline int decode_huffman_brc1(sequential_bit_t *s) {
  if (get_sequential_bit(s)) {
    if (get_sequential_bit(s)) {
      if (get_sequential_bit(s)) {
        if (get_sequential_bit(s)) {
          return 4;
        } else {
          return 3;
        }
      } else {
        return 2;
      }
    } else {
      return 1;
    }
  } else {
    return 0;
  }
}
inline int decode_huffman_brc2(sequential_bit_t *s) {
  if (get_sequential_bit(s)) {
    if (get_sequential_bit(s)) {
      if (get_sequential_bit(s)) {
        if (get_sequential_bit(s)) {
          if (get_sequential_bit(s)) {
            if (get_sequential_bit(s)) {
              return 6;
            } else {
              return 5;
            }
          } else {
            return 4;
          }
        } else {
          return 3;
        }
      } else {
        return 2;
      }
    } else {
      return 1;
    }
  } else {
    return 0;
  }
}
inline int decode_huffman_brc3(sequential_bit_t *s) {
  if (get_sequential_bit(s)) {
    if (get_sequential_bit(s)) {
      if (get_sequential_bit(s)) {
        if (get_sequential_bit(s)) {
          if (get_sequential_bit(s)) {
            if (get_sequential_bit(s)) {
              if (get_sequential_bit(s)) {
                if (get_sequential_bit(s)) {
                  return 9;
                } else {
                  return 8;
                }
              } else {
                return 7;
              }
            } else {
              return 6;
            }
          } else {
            return 5;
          }
        } else {
          return 4;
        }
      } else {
        return 3;
      }
    } else {
      return 2;
    }
  } else {
    if (get_sequential_bit(s)) {
      return 1;
    } else {
      return 0;
    }
  }
}
inline int decode_huffman_brc4(sequential_bit_t *s) {
  if (get_sequential_bit(s)) {
    if (get_sequential_bit(s)) {
      if (get_sequential_bit(s)) {
        if (get_sequential_bit(s)) {
          if (get_sequential_bit(s)) {
            if (get_sequential_bit(s)) {
              if (get_sequential_bit(s)) {
                if (get_sequential_bit(s)) {
                  if (get_sequential_bit(s)) {
                    return 15;
                  } else {
                    return 14;
                  }
                } else {
                  if (get_sequential_bit(s)) {
                    return 13;
                  } else {
                    return 12;
                  }
                }
              } else {
                if (get_sequential_bit(s)) {
                  return 11;
                } else {
                  return 10;
                }
              }
            } else {
              return 9;
            }
          } else {
            return 8;
          }
        } else {
          return 7;
        }
      } else {
        if (get_sequential_bit(s)) {
          return 6;
        } else {
          return 5;
        }
      }
    } else {
      if (get_sequential_bit(s)) {
        return 4;
      } else {
        return 3;
      }
    }
  } else {
    if (get_sequential_bit(s)) {
      if (get_sequential_bit(s)) {
        return 2;
      } else {
        return 1;
      }
    } else {
      return 0;
    }
  }
}
// table 5.2-1 simple reconstruction parameter values B
const std::array<float, 4> table_b0 = {3, 3, (3.16e+0f), (3.53e+0f)};
const std::array<float, 4> table_b1 = {4, 4, (4.08e+0f), (4.3699998e+0f)};
const std::array<float, 6> table_b2 = {6,          6,         6,
                                       (6.15e+0f), (6.5e+0f), (6.88e+0f)};
const std::array<float, 7> table_b3 = {9,          9,         9,         9,
                                       (9.36e+0f), (9.5e+0f), (1.01e+1f)};
const std::array<float, 9> table_b4 = {
    15, 15, 15, 15, 15, 15, (1.522e+1f), (1.55e+1f), (1.6049999e+1f)};
// table 5.2-2 normalized reconstruction levels
const std::array<float, 4> table_nrl0 = {(3.637e-1f), (1.0915001e+0f),
                                         (1.8208e+0f), (2.6406e+0f)};
const std::array<float, 5> table_nrl1 = {(3.042e-1f), (9.127e-1f), (1.5216e+0f),
                                         (2.1313e+0f), (2.8426e+0f)};
const std::array<float, 7> table_nrl2 = {
    (2.305e-1f),  (6.916e-1f),  (1.15279995e+0f), (1.6139999e+0f),
    (2.0754e+0f), (2.5369e+0f), (3.1191e+0f)};
const std::array<float, 10> table_nrl3 = {
    (1.702e-1f),  (5.107e-1f),     (8.511e-1f),  (1.1916e+0f), (1.5321e+0f),
    (1.8726e+0f), (2.2130999e+0f), (2.5536e+0f), (2.8942e+0f), (3.3743998e+0f)};
const std::array<float, 16> table_nrl4 = {
    (1.13e-1f),       (3.389e-1f),  (5.649e-1f),  (7.908e-1f),
    (1.01670004e+0f), (1.2428e+0f), (1.4687e+0f), (1.6947e+0f),
    (1.9206001e+0f),  (2.1466e+0f), (2.3725e+0f), (2.5985e+0f),
    (2.8244e+0f),     (3.0504e+0f), (3.2764e+0f), (3.6623e+0f)};
void init_decode_packet(int packet_idx) {
  auto header = state._header_data[packet_idx].data();
  auto offset = state._header_offset[packet_idx];
  auto number_of_quads =
      ((((0x1) * (header[66]))) + (((0x100) * (((0xFF) & (header[65]))))));
  auto baq_block_length = ((8) * (((1) + (((0xFF) & ((header[38]) >> (0)))))));
  auto number_of_baq_blocks = ((1) + (((((2) * (number_of_quads))) / (256))));
  std::array<uint8_t, 256> brcs;
  std::array<uint8_t, 256> thidxs;
  auto baq_mode = ((0x1F) & ((header[37]) >> (0)));
  auto data = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
  sequential_bit_t s;
  init_sequential_bit_function(
      &s, ((state._header_offset[packet_idx]) + (62) + (6)));
  auto decoded_ie_symbols = 0;
  std::array<float, 65535> decoded_ie_symbols_a;
  // parse ie data
  for (int block = 0; decoded_ie_symbols < number_of_quads; (block)++) {
    auto brc = get_bit_rate_code(&s);
    brcs[block] = brc;
    switch (brc) {
      0 : {
        for (int i = 0; ((i < 128) && (decoded_ie_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc0(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet;
          decoded_ie_symbols_a[decoded_ie_symbols] = v;
          (decoded_ie_symbols)++;
        };
        break;
      }
      1 : {
        for (int i = 0; ((i < 128) && (decoded_ie_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc1(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet;
          decoded_ie_symbols_a[decoded_ie_symbols] = v;
          (decoded_ie_symbols)++;
        };
        break;
      }
      2 : {
        for (int i = 0; ((i < 128) && (decoded_ie_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc2(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet;
          decoded_ie_symbols_a[decoded_ie_symbols] = v;
          (decoded_ie_symbols)++;
        };
        break;
      }
      3 : {
        for (int i = 0; ((i < 128) && (decoded_ie_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc3(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet;
          decoded_ie_symbols_a[decoded_ie_symbols] = v;
          (decoded_ie_symbols)++;
        };
        break;
      }
      4 : {
        for (int i = 0; ((i < 128) && (decoded_ie_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc4(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet;
          decoded_ie_symbols_a[decoded_ie_symbols] = v;
          (decoded_ie_symbols)++;
        };
        break;
      }
    }
  }
  consume_padding_bits(&s);
  auto decoded_io_symbols = 0;
  std::array<float, 65535> decoded_io_symbols_a;
  // parse io data
  for (int block = 0; decoded_io_symbols < number_of_quads; (block)++) {
    auto brc = brcs[block];
    switch (brc) {
      0 : {
        for (int i = 0; ((i < 128) && (decoded_io_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc0(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet;
          decoded_io_symbols_a[decoded_io_symbols] = v;
          (decoded_io_symbols)++;
        };
        break;
      }
      1 : {
        for (int i = 0; ((i < 128) && (decoded_io_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc1(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet;
          decoded_io_symbols_a[decoded_io_symbols] = v;
          (decoded_io_symbols)++;
        };
        break;
      }
      2 : {
        for (int i = 0; ((i < 128) && (decoded_io_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc2(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet;
          decoded_io_symbols_a[decoded_io_symbols] = v;
          (decoded_io_symbols)++;
        };
        break;
      }
      3 : {
        for (int i = 0; ((i < 128) && (decoded_io_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc3(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet;
          decoded_io_symbols_a[decoded_io_symbols] = v;
          (decoded_io_symbols)++;
        };
        break;
      }
      4 : {
        for (int i = 0; ((i < 128) && (decoded_io_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc4(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet;
          decoded_io_symbols_a[decoded_io_symbols] = v;
          (decoded_io_symbols)++;
        };
        break;
      }
    }
  }
  consume_padding_bits(&s);
  auto decoded_qe_symbols = 0;
  std::array<float, 65535> decoded_qe_symbols_a;
  // parse qe data
  for (int block = 0; decoded_qe_symbols < number_of_quads; (block)++) {
    auto thidx = get_threshold_index(&s);
    auto brc = brcs[block];
    thidxs[block] = thidx;
    switch (brc) {
      0 : {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc0(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          if ((thidx) <= (3)) {
            if (mcode < 3) {
              v = ((symbol_sign) * (mcode));
            } else {
              v = ((symbol_sign) * (table_b0[thidx]));
            }
          } else {
            v = ((symbol_sign) * (table_nrl0[mcode]) * (table_sf[thidx]));
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
        break;
      }
      1 : {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc1(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          if ((thidx) <= (3)) {
            if (mcode < 4) {
              v = ((symbol_sign) * (mcode));
            } else {
              v = ((symbol_sign) * (table_b1[thidx]));
            }
          } else {
            v = ((symbol_sign) * (table_nrl1[mcode]) * (table_sf[thidx]));
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
        break;
      }
      2 : {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc2(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          if ((thidx) <= (5)) {
            if (mcode < 6) {
              v = ((symbol_sign) * (mcode));
            } else {
              v = ((symbol_sign) * (table_b2[thidx]));
            }
          } else {
            v = ((symbol_sign) * (table_nrl2[mcode]) * (table_sf[thidx]));
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
        break;
      }
      3 : {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc3(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          if ((thidx) <= (6)) {
            if (mcode < 9) {
              v = ((symbol_sign) * (mcode));
            } else {
              v = ((symbol_sign) * (table_b3[thidx]));
            }
          } else {
            v = ((symbol_sign) * (table_nrl3[mcode]) * (table_sf[thidx]));
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
        break;
      }
      4 : {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc4(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          if ((thidx) <= (8)) {
            if (mcode < 15) {
              v = ((symbol_sign) * (mcode));
            } else {
              v = ((symbol_sign) * (table_b4[thidx]));
            }
          } else {
            v = ((symbol_sign) * (table_nrl4[mcode]) * (table_sf[thidx]));
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
        break;
      }
    }
  }
  consume_padding_bits(&s);
  auto decoded_qo_symbols = 0;
  std::array<float, 65535> decoded_qo_symbols_a;
  // parse qo data
  for (int block = 0; decoded_qo_symbols < number_of_quads; (block)++) {
    auto brc = brcs[block];
    switch (brc) {
      0 : {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc0(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          if ((thidx) <= (3)) {
            if (mcode < 3) {
              v = ((symbol_sign) * (mcode));
            } else {
              v = ((symbol_sign) * (table_b0[thidx]));
            }
          } else {
            v = ((symbol_sign) * (table_nrl0[mcode]) * (table_sf[thidx]));
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
        break;
      }
      1 : {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc1(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          if ((thidx) <= (3)) {
            if (mcode < 4) {
              v = ((symbol_sign) * (mcode));
            } else {
              v = ((symbol_sign) * (table_b1[thidx]));
            }
          } else {
            v = ((symbol_sign) * (table_nrl1[mcode]) * (table_sf[thidx]));
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
        break;
      }
      2 : {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc2(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          if ((thidx) <= (5)) {
            if (mcode < 6) {
              v = ((symbol_sign) * (mcode));
            } else {
              v = ((symbol_sign) * (table_b2[thidx]));
            }
          } else {
            v = ((symbol_sign) * (table_nrl2[mcode]) * (table_sf[thidx]));
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
        break;
      }
      3 : {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc3(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          if ((thidx) <= (6)) {
            if (mcode < 9) {
              v = ((symbol_sign) * (mcode));
            } else {
              v = ((symbol_sign) * (table_b3[thidx]));
            }
          } else {
            v = ((symbol_sign) * (table_nrl3[mcode]) * (table_sf[thidx]));
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
        break;
      }
      4 : {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc4(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          if ((thidx) <= (8)) {
            if (mcode < 15) {
              v = ((symbol_sign) * (mcode));
            } else {
              v = ((symbol_sign) * (table_b4[thidx]));
            }
          } else {
            v = ((symbol_sign) * (table_nrl4[mcode]) * (table_sf[thidx]));
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
        break;
      }
    }
  }
  consume_padding_bits(&s);
};