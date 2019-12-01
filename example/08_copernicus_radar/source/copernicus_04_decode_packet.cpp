
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