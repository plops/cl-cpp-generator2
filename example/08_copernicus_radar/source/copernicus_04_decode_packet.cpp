
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;

uint8_t reverse_bit(uint8_t b) {
  // http://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith64BitsDiv
  // b = ((b * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;
  return ((0xFF) & ((((((((b) * (0x80200802ULL))) & (0x0884422110ULL))) *
                      (0x0101010101ULL))) >>
                    (32)));
}
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
inline int get_bit_rate_code(sequential_bit_t *s) {
  // note: evaluation order is crucial
  return ((((4) * (get_sequential_bit(s)))) +
          (((2) * (get_sequential_bit(s)))) +
          (((1) * (get_sequential_bit(s)))));
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
inline float decode_symbol(sequential_bit_t *s) { return (0.0e+0f); }
void init_decode_packet(int packet_idx) {
  auto header = state._header_data[packet_idx].data();
  auto offset = state._header_offset[packet_idx];
  auto number_of_quads =
      ((((1) * (header[66]))) +
       (((256) * (((0xFF) & ((reverse_bit(header[65])) >> (0)))))));
  auto data = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
  auto baqmod = ((0x1F) & ((reverse_bit(header[37])) >> (3)));
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("") << (" ") << (std::setw(8))
              << (" packet_idx=") << (packet_idx) << (std::setw(8))
              << (" baqmod=") << (baqmod) << (std::endl);
  auto decoded_symbols = 0;
  auto number_of_baq_blocks = ((((2) * (number_of_quads))) / (256));
  sequential_bit_t s;
  std::array<float, 65535> decoded_symbols_a;
  init_sequential_bit_function(
      &s, ((state._header_offset[packet_idx]) + (62) + (6)));
  for (int block = 0; decoded_symbols < number_of_quads;) {
    auto brc = get_bit_rate_code(&s);
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("") << (" ") << (std::setw(8))
                << (" brc=") << (brc) << (std::endl);
    for (int i = 0; ((i < 128) && (decoded_symbols < number_of_quads)); (i)++) {
      auto symbol_sign = (1.e+0f);
      auto symbol = decode_symbol(&s);
      if (get_sequential_bit(&s)) {
        symbol_sign = (-1.e+0f);
      };
      decoded_symbols_a[decoded_symbols] = ((symbol_sign) * (symbol));
      (decoded_symbols)++;
    };
  };
};