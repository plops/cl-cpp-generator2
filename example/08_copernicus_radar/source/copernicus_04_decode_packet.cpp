
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <cassert>
#include <cmath>

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
// table 5.2-3 sigma factors
const std::array<float, 256> table_sf = {
    (0.0e+0f),        (6.3e-1f),        (1.25e+0f),       (1.88e+0f),
    (2.51e+0f),       (3.1300002e+0f),  (3.76e+0f),       (4.3899998e+0f),
    (5.0100005e+0f),  (5.64e+0f),       (6.27e+0f),       (6.89e+0f),
    (7.52e+0f),       (8.1499994e+0f),  (8.7700003e+0f),  (9.3999994e+0f),
    (1.003e+1f),      (1.065e+1f),      (1.12799995e+1f), (1.191e+1f),
    (1.2529999e+1f),  (1.3159999e+1f),  (1.379e+1f),      (1.441e+1f),
    (1.504e+1f),      (1.567e+1f),      (1.6290002e+1f),  (1.692e+1f),
    (1.7549999e+1f),  (1.817e+1f),      (1.88e+1f),       (1.943e+1f),
    (2.005e+1f),      (2.068e+1f),      (2.131e+1f),      (2.193e+1f),
    (2.2559999e+1f),  (2.319e+1f),      (2.3809999e+1f),  (2.4440001e+1f),
    (2.507e+1f),      (2.569e+1f),      (2.6319999e+1f),  (2.6950002e+1f),
    (2.757e+1f),      (2.82e+1f),       (2.883e+1f),      (2.945e+1f),
    (3.008e+1f),      (3.071e+1f),      (3.133e+1f),      (3.196e+1f),
    (3.259e+1f),      (3.321e+1f),      (3.384e+1f),      (3.447e+1f),
    (3.509e+1f),      (3.5720003e+1f),  (3.635e+1f),      (3.697e+1f),
    (3.76e+1f),       (3.823e+1f),      (3.8849998e+1f),  (3.948e+1f),
    (4.011e+1f),      (4.073e+1f),      (4.136e+1f),      (4.1990003e+1f),
    (4.2610002e+1f),  (4.3240002e+1f),  (4.387e+1f),      (4.449e+1f),
    (4.5119998e+1f),  (4.575e+1f),      (4.637e+1f),      (4.7e+1f),
    (4.763e+1f),      (4.825e+1f),      (4.8880002e+1f),  (4.951e+1f),
    (5.0130004e+1f),  (5.076e+1f),      (5.139e+1f),      (5.201e+1f),
    (5.2639997e+1f),  (5.327e+1f),      (5.389e+1f),      (5.452e+1f),
    (5.515e+1f),      (5.577e+1f),      (5.64e+1f),       (5.703e+1f),
    (5.765e+1f),      (5.828e+1f),      (5.891e+1f),      (5.9529996e+1f),
    (6.016e+1f),      (6.079e+1f),      (6.141e+1f),      (6.204e+1f),
    (6.298e+1f),      (6.4239997e+1f),  (6.5489995e+1f),  (6.674e+1f),
    (6.8e+1f),        (6.925e+1f),      (7.05e+1f),       (7.1760005e+1f),
    (7.3010004e+1f),  (7.426e+1f),      (7.5519997e+1f),  (7.6769996e+1f),
    (7.8019994e+1f),  (7.928e+1f),      (8.053e+1f),      (8.178e+1f),
    (8.304e+1f),      (8.4290004e+1f),  (8.554e+1f),      (8.6800003e+1f),
    (8.805e+1f),      (8.93e+1f),       (9.0559995e+1f),  (9.181e+1f),
    (9.306e+1f),      (9.432e+1f),      (9.557e+1f),      (9.6819997e+1f),
    (9.8080003e+1f),  (9.933e+1f),      (1.0058e+2f),     (1.0184e+2f),
    (1.03089996e+2f), (1.04339994e+2f), (1.056e+2f),      (1.0685e+2f),
    (1.081e+2f),      (1.09349996e+2f), (1.1061e+2f),     (1.1186e+2f),
    (1.1311e+2f),     (1.1437e+2f),     (1.1562e+2f),     (1.1687e+2f),
    (1.1813e+2f),     (1.1938e+2f),     (1.20629996e+2f), (1.2189e+2f),
    (1.2314e+2f),     (1.2439e+2f),     (1.2565e+2f),     (1.269e+2f),
    (1.2815e+2f),     (1.2941e+2f),     (1.3066e+2f),     (1.3191e+2f),
    (1.3317e+2f),     (1.3442e+2f),     (1.3566999e+2f),  (1.3692999e+2f),
    (1.3817999e+2f),  (1.3942999e+2f),  (1.4069e+2f),     (1.4194e+2f),
    (1.4319e+2f),     (1.4445e+2f),     (1.457e+2f),      (1.4694999e+2f),
    (1.4821e+2f),     (1.4946e+2f),     (1.5071e+2f),     (1.5197e+2f),
    (1.5322e+2f),     (1.5447e+2f),     (1.5573e+2f),     (1.5698e+2f),
    (1.5822999e+2f),  (1.5949e+2f),     (1.6074e+2f),     (1.6199e+2f),
    (1.6325e+2f),     (1.645e+2f),      (1.6575e+2f),     (1.6701e+2f),
    (1.6826e+2f),     (1.6950999e+2f),  (1.7077e+2f),     (1.7202e+2f),
    (1.7327e+2f),     (1.7453e+2f),     (1.7578e+2f),     (1.7703e+2f),
    (1.7829e+2f),     (1.7954e+2f),     (1.8078999e+2f),  (1.8205e+2f),
    (1.833e+2f),      (1.8455e+2f),     (1.8581e+2f),     (1.8706e+2f),
    (1.8831e+2f),     (1.8957001e+2f),  (1.9082001e+2f),  (1.9207e+2f),
    (1.9333e+2f),     (1.9458e+2f),     (1.9583e+2f),     (1.9709e+2f),
    (1.9834e+2f),     (1.9959e+2f),     (2.0085001e+2f),  (2.0210001e+2f),
    (2.0335001e+2f),  (2.0461e+2f),     (2.0586e+2f),     (2.0711e+2f),
    (2.0837e+2f),     (2.0962e+2f),     (2.1087e+2f),     (2.1213001e+2f),
    (2.1338001e+2f),  (2.1463001e+2f),  (2.1589e+2f),     (2.1714e+2f),
    (2.1839e+2f),     (2.1965e+2f),     (2.209e+2f),      (2.2215e+2f),
    (2.2341001e+2f),  (2.2466001e+2f),  (2.2591001e+2f),  (2.2717e+2f),
    (2.2842e+2f),     (2.2967e+2f),     (2.3092999e+2f),  (2.3218e+2f),
    (2.3343e+2f),     (2.3469e+2f),     (2.3594001e+2f),  (2.3719001e+2f),
    (2.3844999e+2f),  (2.3969999e+2f),  (2.4095e+2f),     (2.4221e+2f),
    (2.4346e+2f),     (2.4471001e+2f),  (2.4597e+2f),     (2.4722e+2f),
    (2.4847001e+2f),  (2.4972999e+2f),  (2.5098e+2f),     (2.5223e+2f),
    (2.5349e+2f),     (2.5474e+2f),     (2.5599e+2f),     (2.5599e+2f)};
int init_decode_packet(
    int packet_idx, int mi_data_delay,
    std::array<std::complex<float>, MAX_NUMBER_QUADS> &output) {
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
  auto number_of_baq_blocks = ((1) + (((((2) * (number_of_quads))) / (256))));
  std::array<uint8_t, 205> brcs;
  std::array<uint8_t, 205> thidxs;
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
  auto data_offset = ((data_delay) - (mi_data_delay));
  auto data = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
  if ((-1) == (mi_data_delay)) {
    data_offset = 0;
  };
  assert((((-1) == (mi_data_delay)) || ((mi_data_delay) <= (data_delay))));
  assert((number_of_baq_blocks) <= (256));
  assert((((0) == (baq_mode)) || ((3) == (baq_mode)) || ((4) == (baq_mode)) ||
          ((5) == (baq_mode)) || ((12) == (baq_mode)) || ((13) == (baq_mode)) ||
          ((14) == (baq_mode))));
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("") << (" ") << (std::setw(8))
              << (" packet_idx=") << (packet_idx) << (std::setw(8))
              << (" baq_mode=") << (baq_mode) << (std::setw(8))
              << (" data_delay_us=") << (data_delay_us) << (std::setw(8))
              << (" data_delay=") << (data_delay) << (std::setw(8))
              << (" number_of_quads=") << (number_of_quads) << (std::endl);
  sequential_bit_t s;
  init_sequential_bit_function(
      &s, ((state._header_offset[packet_idx]) + (62) + (6)));
  auto decoded_ie_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_ie_symbols_a;
  // parse ie data
  for (int block = 0; decoded_ie_symbols < number_of_quads; (block)++) {
    auto brc = get_bit_rate_code(&s);
    brcs[block] = brc;
    switch (brc) {
    case 0: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      for (int i = 0; ((i < 128) && (decoded_ie_symbols < number_of_quads));
           (i)++) {
        auto sign_bit = get_sequential_bit(&s);
        auto mcode = decode_huffman_brc0(&s);
        auto symbol_sign = (1.e+0f);
        if (sign_bit) {
          symbol_sign = (-1.e+0f);
        };
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later;
        decoded_ie_symbols_a[decoded_ie_symbols] = v;
        (decoded_ie_symbols)++;
      };
      break;
    }
    case 1: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      for (int i = 0; ((i < 128) && (decoded_ie_symbols < number_of_quads));
           (i)++) {
        auto sign_bit = get_sequential_bit(&s);
        auto mcode = decode_huffman_brc1(&s);
        auto symbol_sign = (1.e+0f);
        if (sign_bit) {
          symbol_sign = (-1.e+0f);
        };
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later;
        decoded_ie_symbols_a[decoded_ie_symbols] = v;
        (decoded_ie_symbols)++;
      };
      break;
    }
    case 2: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      for (int i = 0; ((i < 128) && (decoded_ie_symbols < number_of_quads));
           (i)++) {
        auto sign_bit = get_sequential_bit(&s);
        auto mcode = decode_huffman_brc2(&s);
        auto symbol_sign = (1.e+0f);
        if (sign_bit) {
          symbol_sign = (-1.e+0f);
        };
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later;
        decoded_ie_symbols_a[decoded_ie_symbols] = v;
        (decoded_ie_symbols)++;
      };
      break;
    }
    case 3: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      for (int i = 0; ((i < 128) && (decoded_ie_symbols < number_of_quads));
           (i)++) {
        auto sign_bit = get_sequential_bit(&s);
        auto mcode = decode_huffman_brc3(&s);
        auto symbol_sign = (1.e+0f);
        if (sign_bit) {
          symbol_sign = (-1.e+0f);
        };
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later;
        decoded_ie_symbols_a[decoded_ie_symbols] = v;
        (decoded_ie_symbols)++;
      };
      break;
    }
    case 4: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      for (int i = 0; ((i < 128) && (decoded_ie_symbols < number_of_quads));
           (i)++) {
        auto sign_bit = get_sequential_bit(&s);
        auto mcode = decode_huffman_brc4(&s);
        auto symbol_sign = (1.e+0f);
        if (sign_bit) {
          symbol_sign = (-1.e+0f);
        };
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later;
        decoded_ie_symbols_a[decoded_ie_symbols] = v;
        (decoded_ie_symbols)++;
      };
      break;
    }
    }
  }
  consume_padding_bits(&s);
  auto decoded_io_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_io_symbols_a;
  // parse io data
  for (int block = 0; decoded_io_symbols < number_of_quads; (block)++) {
    auto brc = brcs[block];
    switch (brc) {
    case 0: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      for (int i = 0; ((i < 128) && (decoded_io_symbols < number_of_quads));
           (i)++) {
        auto sign_bit = get_sequential_bit(&s);
        auto mcode = decode_huffman_brc0(&s);
        auto symbol_sign = (1.e+0f);
        if (sign_bit) {
          symbol_sign = (-1.e+0f);
        };
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later;
        decoded_io_symbols_a[decoded_io_symbols] = v;
        (decoded_io_symbols)++;
      };
      break;
    }
    case 1: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      for (int i = 0; ((i < 128) && (decoded_io_symbols < number_of_quads));
           (i)++) {
        auto sign_bit = get_sequential_bit(&s);
        auto mcode = decode_huffman_brc1(&s);
        auto symbol_sign = (1.e+0f);
        if (sign_bit) {
          symbol_sign = (-1.e+0f);
        };
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later;
        decoded_io_symbols_a[decoded_io_symbols] = v;
        (decoded_io_symbols)++;
      };
      break;
    }
    case 2: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      for (int i = 0; ((i < 128) && (decoded_io_symbols < number_of_quads));
           (i)++) {
        auto sign_bit = get_sequential_bit(&s);
        auto mcode = decode_huffman_brc2(&s);
        auto symbol_sign = (1.e+0f);
        if (sign_bit) {
          symbol_sign = (-1.e+0f);
        };
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later;
        decoded_io_symbols_a[decoded_io_symbols] = v;
        (decoded_io_symbols)++;
      };
      break;
    }
    case 3: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      for (int i = 0; ((i < 128) && (decoded_io_symbols < number_of_quads));
           (i)++) {
        auto sign_bit = get_sequential_bit(&s);
        auto mcode = decode_huffman_brc3(&s);
        auto symbol_sign = (1.e+0f);
        if (sign_bit) {
          symbol_sign = (-1.e+0f);
        };
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later;
        decoded_io_symbols_a[decoded_io_symbols] = v;
        (decoded_io_symbols)++;
      };
      break;
    }
    case 4: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      for (int i = 0; ((i < 128) && (decoded_io_symbols < number_of_quads));
           (i)++) {
        auto sign_bit = get_sequential_bit(&s);
        auto mcode = decode_huffman_brc4(&s);
        auto symbol_sign = (1.e+0f);
        if (sign_bit) {
          symbol_sign = (-1.e+0f);
        };
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later;
        decoded_io_symbols_a[decoded_io_symbols] = v;
        (decoded_io_symbols)++;
      };
      break;
    }
    }
  }
  consume_padding_bits(&s);
  auto decoded_qe_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_qe_symbols_a;
  // parse qe data
  for (int block = 0; decoded_qe_symbols < number_of_quads; (block)++) {
    auto thidx = get_threshold_index(&s);
    auto brc = brcs[block];
    thidxs[block] = thidx;
    switch (brc) {
    case 0: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (3)) {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc0(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          auto v = (0.0e+0f);
          if (mcode < 3) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b0.at(thidx)));
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
      } else {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc0(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl0.at(mcode)) * (table_sf.at(thidx)));
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
      }
      break;
    }
    case 1: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (3)) {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc1(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          auto v = (0.0e+0f);
          if (mcode < 4) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b1.at(thidx)));
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
      } else {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc1(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl1.at(mcode)) * (table_sf.at(thidx)));
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
      }
      break;
    }
    case 2: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (5)) {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc2(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          auto v = (0.0e+0f);
          if (mcode < 6) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b2.at(thidx)));
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
      } else {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc2(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl2.at(mcode)) * (table_sf.at(thidx)));
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
      }
      break;
    }
    case 3: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (6)) {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc3(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          auto v = (0.0e+0f);
          if (mcode < 9) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b3.at(thidx)));
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
      } else {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc3(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl3.at(mcode)) * (table_sf.at(thidx)));
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
      }
      break;
    }
    case 4: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (8)) {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc4(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          auto v = (0.0e+0f);
          if (mcode < 15) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b4.at(thidx)));
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
      } else {
        for (int i = 0; ((i < 128) && (decoded_qe_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc4(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qe p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl4.at(mcode)) * (table_sf.at(thidx)));
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        };
      }
      break;
    }
    }
  }
  consume_padding_bits(&s);
  auto decoded_qo_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_qo_symbols_a;
  // parse qo data
  for (int block = 0; decoded_qo_symbols < number_of_quads; (block)++) {
    auto brc = brcs[block];
    auto thidx = thidxs[block];
    switch (brc) {
    case 0: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (3)) {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc0(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          auto v = (0.0e+0f);
          if (mcode < 3) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b0.at(thidx)));
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
      } else {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc0(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl0.at(mcode)) * (table_sf.at(thidx)));
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
      }
      break;
    }
    case 1: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (3)) {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc1(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          auto v = (0.0e+0f);
          if (mcode < 4) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b1.at(thidx)));
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
      } else {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc1(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl1.at(mcode)) * (table_sf.at(thidx)));
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
      }
      break;
    }
    case 2: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (5)) {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc2(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          auto v = (0.0e+0f);
          if (mcode < 6) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b2.at(thidx)));
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
      } else {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc2(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl2.at(mcode)) * (table_sf.at(thidx)));
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
      }
      break;
    }
    case 3: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (6)) {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc3(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          auto v = (0.0e+0f);
          if (mcode < 9) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b3.at(thidx)));
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
      } else {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc3(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl3.at(mcode)) * (table_sf.at(thidx)));
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
      }
      break;
    }
    case 4: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (8)) {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc4(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          auto v = (0.0e+0f);
          if (mcode < 15) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b4.at(thidx)));
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
      } else {
        for (int i = 0; ((i < 128) && (decoded_qo_symbols < number_of_quads));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc4(&s);
          auto symbol_sign = (1.e+0f);
          if (sign_bit) {
            symbol_sign = (-1.e+0f);
          };
          // decode qo p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl4.at(mcode)) * (table_sf.at(thidx)));
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        };
      }
      break;
    }
    }
  }
  consume_padding_bits(&s);
  for (int block = 0; block < number_of_baq_blocks; (block) += (1)) {
    auto brc = brcs[block];
    auto thidx = thidxs[block];
    switch (brc) {
    case 0: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (3)) {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_ie_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode ie p.75
          auto v = (0.0e+0f);
          if (mcode < 3) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b0.at(thidx)));
          };
          decoded_ie_symbols_a[pos] = v;
        };
      } else {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_ie_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode ie p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl0.at(mcode)) * (table_sf.at(thidx)));
          decoded_ie_symbols_a[pos] = v;
        };
      }
      break;
    }
    case 1: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (3)) {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_ie_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode ie p.75
          auto v = (0.0e+0f);
          if (mcode < 4) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b1.at(thidx)));
          };
          decoded_ie_symbols_a[pos] = v;
        };
      } else {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_ie_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode ie p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl1.at(mcode)) * (table_sf.at(thidx)));
          decoded_ie_symbols_a[pos] = v;
        };
      }
      break;
    }
    case 2: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (5)) {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_ie_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode ie p.75
          auto v = (0.0e+0f);
          if (mcode < 6) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b2.at(thidx)));
          };
          decoded_ie_symbols_a[pos] = v;
        };
      } else {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_ie_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode ie p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl2.at(mcode)) * (table_sf.at(thidx)));
          decoded_ie_symbols_a[pos] = v;
        };
      }
      break;
    }
    case 3: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (6)) {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_ie_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode ie p.75
          auto v = (0.0e+0f);
          if (mcode < 9) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b3.at(thidx)));
          };
          decoded_ie_symbols_a[pos] = v;
        };
      } else {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_ie_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode ie p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl3.at(mcode)) * (table_sf.at(thidx)));
          decoded_ie_symbols_a[pos] = v;
        };
      }
      break;
    }
    case 4: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (8)) {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_ie_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode ie p.75
          auto v = (0.0e+0f);
          if (mcode < 15) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b4.at(thidx)));
          };
          decoded_ie_symbols_a[pos] = v;
        };
      } else {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_ie_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode ie p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl4.at(mcode)) * (table_sf.at(thidx)));
          decoded_ie_symbols_a[pos] = v;
        };
      }
      break;
    }
    }
    switch (brc) {
    case 0: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (3)) {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_io_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode io p.75
          auto v = (0.0e+0f);
          if (mcode < 3) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b0.at(thidx)));
          };
          decoded_io_symbols_a[pos] = v;
        };
      } else {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_io_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode io p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl0.at(mcode)) * (table_sf.at(thidx)));
          decoded_io_symbols_a[pos] = v;
        };
      }
      break;
    }
    case 1: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (3)) {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_io_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode io p.75
          auto v = (0.0e+0f);
          if (mcode < 4) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b1.at(thidx)));
          };
          decoded_io_symbols_a[pos] = v;
        };
      } else {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_io_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode io p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl1.at(mcode)) * (table_sf.at(thidx)));
          decoded_io_symbols_a[pos] = v;
        };
      }
      break;
    }
    case 2: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (5)) {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_io_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode io p.75
          auto v = (0.0e+0f);
          if (mcode < 6) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b2.at(thidx)));
          };
          decoded_io_symbols_a[pos] = v;
        };
      } else {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_io_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode io p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl2.at(mcode)) * (table_sf.at(thidx)));
          decoded_io_symbols_a[pos] = v;
        };
      }
      break;
    }
    case 3: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (6)) {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_io_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode io p.75
          auto v = (0.0e+0f);
          if (mcode < 9) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b3.at(thidx)));
          };
          decoded_io_symbols_a[pos] = v;
        };
      } else {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_io_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode io p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl3.at(mcode)) * (table_sf.at(thidx)));
          decoded_io_symbols_a[pos] = v;
        };
      }
      break;
    }
    case 4: {
      if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) ||
             ((3) == (brc)) || ((4) == (brc))))) {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error: out of range") << (" ")
                    << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
        assert(0);
      };
      if ((thidx) <= (8)) {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_io_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode io p.75
          auto v = (0.0e+0f);
          if (mcode < 15) {
            v = ((symbol_sign) * (mcode));
          } else {
            v = ((symbol_sign) * (table_b4.at(thidx)));
          };
          decoded_io_symbols_a[pos] = v;
        };
      } else {
        for (int i = 0; i < 128; (i) += (1)) {
          auto pos = ((i) + (((128) * (block))));
          auto scode = decoded_io_symbols_a[pos];
          auto mcode = static_cast<int>(fabsf(scode));
          auto symbol_sign = copysignf((1.e+0f), scode);
          // decode io p.75
          auto v = (0.0e+0f);
          v = ((symbol_sign) * (table_nrl4.at(mcode)) * (table_sf.at(thidx)));
          decoded_io_symbols_a[pos] = v;
        };
      }
      break;
    }
    };
  }
  for (int i = 0; i < decoded_ie_symbols; (i) += (1)) {
    output[((data_offset) + (((2) * (i))))].real(decoded_ie_symbols_a[i]);
    output[((data_offset) + (((2) * (i))))].imag(decoded_qe_symbols_a[i]);
    output[((data_offset) + (((1) + (((2) * (i))))))].real(
        decoded_io_symbols_a[i]);
    output[((data_offset) + (((1) + (((2) * (i))))))].imag(
        decoded_qo_symbols_a[i]);
  }
  auto n = ((decoded_ie_symbols) + (decoded_io_symbols));
  return n;
};