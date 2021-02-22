
#include "utils.h"

#include "globals.h"

#include "proto2.h"

extern State state;
#include <cassert>
#include <cmath>

void init_sequential_bit_function(sequential_bit_t *seq_state,
                                  size_t byte_pos) {
  seq_state->data = &(static_cast<uint8_t *>(state._mmap_data)[byte_pos]);
  seq_state->current_bit_count = 0;
}

void consume_padding_bits(sequential_bit_t *s) {
  auto byte_offset = static_cast<int>(
      ((s->data) - (static_cast<uint8_t *>(state._mmap_data))));
  // make sure we are at first bit of an even byte in the next read
  if ((0) == (byte_offset % 2)) {
    // we are in an even byte
    if ((0) == (s->current_bit_count)) {
      // nothing to be done
    } else {
      (s->data) += (2);
      s->current_bit_count = 0;
    }
  } else {
    // we are in an odd byte
    (s->data) += (1);
    s->current_bit_count = 0;
  }
}
inline int get_bit_rate_code(sequential_bit_t *s) {
  // note: evaluation order is crucial
  auto brc = ((((0x4) * (get_sequential_bit(s)))) +
              (((0x2) * (get_sequential_bit(s)))) +
              (((0x1) * (get_sequential_bit(s)))));
  if (!((((0) == (brc)) || ((1) == (brc)) || ((2) == (brc)) || ((3) == (brc)) ||
         ((4) == (brc))))) {
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("brc out of range") << (" ")
                << (std::setw(8)) << (" s->current_bit_count=")
                << (s->current_bit_count) << (std::setw(8))
                << (" ((s->data)-(static_cast<uint8_t*>(state._mmap_data)))=")
                << (((s->data) - (static_cast<uint8_t *>(state._mmap_data))))
                << (std::setw(8)) << (" brc=") << (brc) << (std::endl);
    throw std::out_of_range("brc");
  }
  return brc;
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
const std::array<const float, 4> table_b0 = {3, 3, (3.160f), (3.530f)};
const std::array<const float, 4> table_b1 = {4, 4, (4.080f), (4.370f)};
const std::array<const float, 6> table_b2 = {6,        6,       6,
                                             (6.150f), (6.50f), (6.880f)};
const std::array<const float, 7> table_b3 = {9,        9,       9,       9,
                                             (9.360f), (9.50f), (10.10f)};
const std::array<const float, 9> table_b4 = {
    15, 15, 15, 15, 15, 15, (15.220f), (15.50f), (16.050f)};
// table 5.2-2 normalized reconstruction levels
const std::array<const float, 4> table_nrl0 = {(0.36370f), (1.09150f),
                                               (1.82080f), (2.64060f)};
const std::array<const float, 5> table_nrl1 = {
    (0.30420f), (0.91270f), (1.52160f), (2.13130f), (2.84260f)};
const std::array<const float, 7> table_nrl2 = {
    (0.23050f), (0.69160f), (1.15280f), (1.6140f),
    (2.07540f), (2.53690f), (3.11910f)};
const std::array<const float, 10> table_nrl3 = {
    (0.17020f), (0.51070f), (0.85110f), (1.19160f), (1.53210f),
    (1.87260f), (2.21310f), (2.55360f), (2.89420f), (3.37440f)};
const std::array<const float, 16> table_nrl4 = {
    (0.1130f),  (0.33890f), (0.56490f), (0.79080f), (1.01670f), (1.24280f),
    (1.46870f), (1.69470f), (1.92060f), (2.14660f), (2.37250f), (2.59850f),
    (2.82440f), (3.05040f), (3.27640f), (3.66230f)};
// table 5.2-3 sigma factors
extern const std::array<const float, 256> table_sf = {
    (0.f),      (0.630f),   (1.250f),   (1.880f),   (2.510f),   (3.130f),
    (3.760f),   (4.390f),   (5.010f),   (5.640f),   (6.270f),   (6.890f),
    (7.520f),   (8.150f),   (8.770f),   (9.40f),    (10.030f),  (10.650f),
    (11.280f),  (11.910f),  (12.530f),  (13.160f),  (13.790f),  (14.410f),
    (15.040f),  (15.670f),  (16.290f),  (16.920f),  (17.550f),  (18.170f),
    (18.80f),   (19.430f),  (20.050f),  (20.680f),  (21.310f),  (21.930f),
    (22.560f),  (23.190f),  (23.810f),  (24.440f),  (25.070f),  (25.690f),
    (26.320f),  (26.950f),  (27.570f),  (28.20f),   (28.830f),  (29.450f),
    (30.080f),  (30.710f),  (31.330f),  (31.960f),  (32.590f),  (33.210f),
    (33.840f),  (34.470f),  (35.090f),  (35.720f),  (36.350f),  (36.970f),
    (37.60f),   (38.230f),  (38.850f),  (39.480f),  (40.110f),  (40.730f),
    (41.360f),  (41.990f),  (42.610f),  (43.240f),  (43.870f),  (44.490f),
    (45.120f),  (45.750f),  (46.370f),  (47.f),     (47.630f),  (48.250f),
    (48.880f),  (49.510f),  (50.130f),  (50.760f),  (51.390f),  (52.010f),
    (52.640f),  (53.270f),  (53.890f),  (54.520f),  (55.150f),  (55.770f),
    (56.40f),   (57.030f),  (57.650f),  (58.280f),  (58.910f),  (59.530f),
    (60.160f),  (60.790f),  (61.410f),  (62.040f),  (62.980f),  (64.240f),
    (65.490f),  (66.740f),  (68.f),     (69.250f),  (70.50f),   (71.760f),
    (73.010f),  (74.260f),  (75.520f),  (76.770f),  (78.020f),  (79.280f),
    (80.530f),  (81.780f),  (83.040f),  (84.290f),  (85.540f),  (86.80f),
    (88.050f),  (89.30f),   (90.560f),  (91.810f),  (93.060f),  (94.320f),
    (95.570f),  (96.820f),  (98.080f),  (99.330f),  (100.580f), (101.840f),
    (103.090f), (104.340f), (105.60f),  (106.850f), (108.10f),  (109.350f),
    (110.610f), (111.860f), (113.110f), (114.370f), (115.620f), (116.870f),
    (118.130f), (119.380f), (120.630f), (121.890f), (123.140f), (124.390f),
    (125.650f), (126.90f),  (128.150f), (129.410f), (130.660f), (131.910f),
    (133.170f), (134.420f), (135.670f), (136.930f), (138.180f), (139.430f),
    (140.690f), (141.940f), (143.190f), (144.450f), (145.70f),  (146.950f),
    (148.210f), (149.460f), (150.710f), (151.970f), (153.220f), (154.470f),
    (155.730f), (156.980f), (158.230f), (159.490f), (160.740f), (161.990f),
    (163.250f), (164.50f),  (165.750f), (167.010f), (168.260f), (169.510f),
    (170.770f), (172.020f), (173.270f), (174.530f), (175.780f), (177.030f),
    (178.290f), (179.540f), (180.790f), (182.050f), (183.30f),  (184.550f),
    (185.810f), (187.060f), (188.310f), (189.570f), (190.820f), (192.070f),
    (193.330f), (194.580f), (195.830f), (197.090f), (198.340f), (199.590f),
    (200.850f), (202.10f),  (203.350f), (204.610f), (205.860f), (207.110f),
    (208.370f), (209.620f), (210.870f), (212.130f), (213.380f), (214.630f),
    (215.890f), (217.140f), (218.390f), (219.650f), (220.90f),  (222.150f),
    (223.410f), (224.660f), (225.910f), (227.170f), (228.420f), (229.670f),
    (230.930f), (232.180f), (233.430f), (234.690f), (235.940f), (237.190f),
    (238.450f), (239.70f),  (240.950f), (242.210f), (243.460f), (244.710f),
    (245.970f), (247.220f), (248.470f), (249.730f), (250.980f), (252.230f),
    (253.490f), (254.740f), (255.990f), (255.990f)};
int init_decode_packet(int packet_idx, std::complex<float> *output) {
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
  auto number_of_baq_blocks =
      static_cast<int>(round(ceil((((((2.0f)) * (number_of_quads))) / (256)))));
  std::array<uint8_t, 205> brcs;
  std::array<uint8_t, 205> thidxs;
  auto baq_mode = ((0x1F) & ((header[37]) >> (0)));
  auto fref = (37.53472f);
  auto swst = ((((((0x1) * (header[55]))) + (((0x100) * (header[54]))) +
                 (((0x10000) * (((0xFF) & (header[53]))))))) /
               (fref));
  auto delta_t_suppressed = (((3.20e+2)) / (((8) * (fref))));
  auto data_delay_us = ((swst) + (delta_t_suppressed));
  auto data_delay =
      ((40) + (((((0x1) * (header[55]))) + (((0x100) * (header[54]))) +
                (((0x10000) * (((0xFF) & (header[53]))))))));
  auto data = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
  assert((number_of_baq_blocks) <= (256));
  assert((((0) == (baq_mode)) || ((3) == (baq_mode)) || ((4) == (baq_mode)) ||
          ((5) == (baq_mode)) || ((12) == (baq_mode)) || ((13) == (baq_mode)) ||
          ((14) == (baq_mode))));
  sequential_bit_t s;
  init_sequential_bit_function(
      &s, ((state._header_offset[packet_idx]) + (62) + (6)));
  auto decoded_ie_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_ie_symbols_a;
  for (auto i = 0; (i) < (MAX_NUMBER_QUADS); (i) += (1)) {
    decoded_ie_symbols_a[i] = (0.f);
  }
  // parse ie data
  for (int block = 0; (decoded_ie_symbols) < (number_of_quads); (block)++) {
    auto brc = get_bit_rate_code(&s);
    brcs[block] = brc;
    switch (brc) {
    case 0: {
      {

        // reconstruction law block=ie thidx-choice=thidx-unknown brc=0
        for (int i = 0;
             (((i) < (128)) && ((decoded_ie_symbols) < (number_of_quads)));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc0(&s);
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet, will be processed later
          decoded_ie_symbols_a[decoded_ie_symbols] = v;
          (decoded_ie_symbols)++;
        }
        break;
      }
      break;
    }
    case 1: {
      {

        // reconstruction law block=ie thidx-choice=thidx-unknown brc=1
        for (int i = 0;
             (((i) < (128)) && ((decoded_ie_symbols) < (number_of_quads)));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc1(&s);
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet, will be processed later
          decoded_ie_symbols_a[decoded_ie_symbols] = v;
          (decoded_ie_symbols)++;
        }
        break;
      }
      break;
    }
    case 2: {
      {

        // reconstruction law block=ie thidx-choice=thidx-unknown brc=2
        for (int i = 0;
             (((i) < (128)) && ((decoded_ie_symbols) < (number_of_quads)));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc2(&s);
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet, will be processed later
          decoded_ie_symbols_a[decoded_ie_symbols] = v;
          (decoded_ie_symbols)++;
        }
        break;
      }
      break;
    }
    case 3: {
      {

        // reconstruction law block=ie thidx-choice=thidx-unknown brc=3
        for (int i = 0;
             (((i) < (128)) && ((decoded_ie_symbols) < (number_of_quads)));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc3(&s);
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet, will be processed later
          decoded_ie_symbols_a[decoded_ie_symbols] = v;
          (decoded_ie_symbols)++;
        }
        break;
      }
      break;
    }
    case 4: {
      {

        // reconstruction law block=ie thidx-choice=thidx-unknown brc=4
        for (int i = 0;
             (((i) < (128)) && ((decoded_ie_symbols) < (number_of_quads)));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc4(&s);
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet, will be processed later
          decoded_ie_symbols_a[decoded_ie_symbols] = v;
          (decoded_ie_symbols)++;
        }
        break;
      }
      break;
    }
    default: {
      {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error brc out of range")
                    << (" ") << (std::setw(8)) << (" brc=") << (brc)
                    << (std::endl);
        assert(0);
        break;
      }
      break;
    }
    }
  }
  consume_padding_bits(&s);
  auto decoded_io_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_io_symbols_a;
  for (auto i = 0; (i) < (MAX_NUMBER_QUADS); (i) += (1)) {
    decoded_io_symbols_a[i] = (0.f);
  }
  // parse io data
  for (int block = 0; (decoded_io_symbols) < (number_of_quads); (block)++) {
    auto brc = brcs[block];
    switch (brc) {
    case 0: {
      {

        // reconstruction law block=io thidx-choice=thidx-unknown brc=0
        for (int i = 0;
             (((i) < (128)) && ((decoded_io_symbols) < (number_of_quads)));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc0(&s);
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet, will be processed later
          decoded_io_symbols_a[decoded_io_symbols] = v;
          (decoded_io_symbols)++;
        }
        break;
      }
      break;
    }
    case 1: {
      {

        // reconstruction law block=io thidx-choice=thidx-unknown brc=1
        for (int i = 0;
             (((i) < (128)) && ((decoded_io_symbols) < (number_of_quads)));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc1(&s);
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet, will be processed later
          decoded_io_symbols_a[decoded_io_symbols] = v;
          (decoded_io_symbols)++;
        }
        break;
      }
      break;
    }
    case 2: {
      {

        // reconstruction law block=io thidx-choice=thidx-unknown brc=2
        for (int i = 0;
             (((i) < (128)) && ((decoded_io_symbols) < (number_of_quads)));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc2(&s);
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet, will be processed later
          decoded_io_symbols_a[decoded_io_symbols] = v;
          (decoded_io_symbols)++;
        }
        break;
      }
      break;
    }
    case 3: {
      {

        // reconstruction law block=io thidx-choice=thidx-unknown brc=3
        for (int i = 0;
             (((i) < (128)) && ((decoded_io_symbols) < (number_of_quads)));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc3(&s);
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet, will be processed later
          decoded_io_symbols_a[decoded_io_symbols] = v;
          (decoded_io_symbols)++;
        }
        break;
      }
      break;
    }
    case 4: {
      {

        // reconstruction law block=io thidx-choice=thidx-unknown brc=4
        for (int i = 0;
             (((i) < (128)) && ((decoded_io_symbols) < (number_of_quads)));
             (i)++) {
          auto sign_bit = get_sequential_bit(&s);
          auto mcode = decode_huffman_brc4(&s);
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          auto v = ((symbol_sign) * (mcode));
          // in ie and io we don't have thidx yet, will be processed later
          decoded_io_symbols_a[decoded_io_symbols] = v;
          (decoded_io_symbols)++;
        }
        break;
      }
      break;
    }
    default: {
      {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error brc out of range")
                    << (" ") << (std::setw(8)) << (" brc=") << (brc)
                    << (std::endl);
        assert(0);
        break;
      }
      break;
    }
    }
  }
  consume_padding_bits(&s);
  auto decoded_qe_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_qe_symbols_a;
  for (auto i = 0; (i) < (MAX_NUMBER_QUADS); (i) += (1)) {
    decoded_qe_symbols_a[i] = (0.f);
  }
  // parse qe data
  for (int block = 0; (decoded_qe_symbols) < (number_of_quads); (block)++) {
    auto thidx = get_threshold_index(&s);
    auto brc = brcs[block];
    thidxs[block] = thidx;
    switch (brc) {
    case 0: {
      {

        if ((thidx) <= (3)) {
          // reconstruction law block=qe thidx-choice=simple brc=0
          for (int i = 0;
               (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc0(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qe p.75
            auto v = (0.f);
            try {
              if ((mcode) < (3)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (3)) {
                  v = ((symbol_sign) * (table_b0.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("exception simple brc=0")
                          << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qe_symbols_a[decoded_qe_symbols] = v;
            (decoded_qe_symbols)++;
          }
        } else {
          // reconstruction law block=qe thidx-choice=normal brc=0
          for (int i = 0;
               (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc0(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qe p.75
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl0.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf brc=0") << (" ")
                          << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qe_symbols_a[decoded_qe_symbols] = v;
            (decoded_qe_symbols)++;
          }
        }
        break;
      }
      break;
    }
    case 1: {
      {

        if ((thidx) <= (3)) {
          // reconstruction law block=qe thidx-choice=simple brc=1
          for (int i = 0;
               (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc1(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qe p.75
            auto v = (0.f);
            try {
              if ((mcode) < (4)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (4)) {
                  v = ((symbol_sign) * (table_b1.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("exception simple brc=1")
                          << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qe_symbols_a[decoded_qe_symbols] = v;
            (decoded_qe_symbols)++;
          }
        } else {
          // reconstruction law block=qe thidx-choice=normal brc=1
          for (int i = 0;
               (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc1(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qe p.75
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl1.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf brc=1") << (" ")
                          << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qe_symbols_a[decoded_qe_symbols] = v;
            (decoded_qe_symbols)++;
          }
        }
        break;
      }
      break;
    }
    case 2: {
      {

        if ((thidx) <= (5)) {
          // reconstruction law block=qe thidx-choice=simple brc=2
          for (int i = 0;
               (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc2(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qe p.75
            auto v = (0.f);
            try {
              if ((mcode) < (6)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (6)) {
                  v = ((symbol_sign) * (table_b2.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("exception simple brc=2")
                          << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qe_symbols_a[decoded_qe_symbols] = v;
            (decoded_qe_symbols)++;
          }
        } else {
          // reconstruction law block=qe thidx-choice=normal brc=2
          for (int i = 0;
               (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc2(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qe p.75
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl2.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf brc=2") << (" ")
                          << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qe_symbols_a[decoded_qe_symbols] = v;
            (decoded_qe_symbols)++;
          }
        }
        break;
      }
      break;
    }
    case 3: {
      {

        if ((thidx) <= (6)) {
          // reconstruction law block=qe thidx-choice=simple brc=3
          for (int i = 0;
               (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc3(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qe p.75
            auto v = (0.f);
            try {
              if ((mcode) < (9)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (9)) {
                  v = ((symbol_sign) * (table_b3.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("exception simple brc=3")
                          << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qe_symbols_a[decoded_qe_symbols] = v;
            (decoded_qe_symbols)++;
          }
        } else {
          // reconstruction law block=qe thidx-choice=normal brc=3
          for (int i = 0;
               (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc3(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qe p.75
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl3.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf brc=3") << (" ")
                          << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qe_symbols_a[decoded_qe_symbols] = v;
            (decoded_qe_symbols)++;
          }
        }
        break;
      }
      break;
    }
    case 4: {
      {

        if ((thidx) <= (8)) {
          // reconstruction law block=qe thidx-choice=simple brc=4
          for (int i = 0;
               (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc4(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qe p.75
            auto v = (0.f);
            try {
              if ((mcode) < (15)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (15)) {
                  v = ((symbol_sign) * (table_b4.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("exception simple brc=4")
                          << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qe_symbols_a[decoded_qe_symbols] = v;
            (decoded_qe_symbols)++;
          }
        } else {
          // reconstruction law block=qe thidx-choice=normal brc=4
          for (int i = 0;
               (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc4(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qe p.75
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl4.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf brc=4") << (" ")
                          << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qe_symbols_a[decoded_qe_symbols] = v;
            (decoded_qe_symbols)++;
          }
        }
        break;
      }
      break;
    }
    default: {
      {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error brc out of range")
                    << (" ") << (std::setw(8)) << (" brc=") << (brc)
                    << (std::endl);
        assert(0);
        break;
      }
      break;
    }
    }
  }
  consume_padding_bits(&s);
  auto decoded_qo_symbols = 0;
  std::array<float, MAX_NUMBER_QUADS> decoded_qo_symbols_a;
  for (auto i = 0; (i) < (MAX_NUMBER_QUADS); (i) += (1)) {
    decoded_qo_symbols_a[i] = (0.f);
  }
  // parse qo data
  for (int block = 0; (decoded_qo_symbols) < (number_of_quads); (block)++) {
    auto brc = brcs[block];
    auto thidx = thidxs[block];
    switch (brc) {
    case 0: {
      {

        if ((thidx) <= (3)) {
          // reconstruction law block=qo thidx-choice=simple brc=0
          for (int i = 0;
               (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc0(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qo p.75
            auto v = (0.f);
            try {
              if ((mcode) < (3)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (3)) {
                  v = ((symbol_sign) * (table_b0.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("exception simple brc=0")
                          << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qo_symbols_a[decoded_qo_symbols] = v;
            (decoded_qo_symbols)++;
          }
        } else {
          // reconstruction law block=qo thidx-choice=normal brc=0
          for (int i = 0;
               (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc0(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qo p.75
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl0.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf brc=0") << (" ")
                          << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qo_symbols_a[decoded_qo_symbols] = v;
            (decoded_qo_symbols)++;
          }
        }
        break;
      }
      break;
    }
    case 1: {
      {

        if ((thidx) <= (3)) {
          // reconstruction law block=qo thidx-choice=simple brc=1
          for (int i = 0;
               (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc1(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qo p.75
            auto v = (0.f);
            try {
              if ((mcode) < (4)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (4)) {
                  v = ((symbol_sign) * (table_b1.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("exception simple brc=1")
                          << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qo_symbols_a[decoded_qo_symbols] = v;
            (decoded_qo_symbols)++;
          }
        } else {
          // reconstruction law block=qo thidx-choice=normal brc=1
          for (int i = 0;
               (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc1(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qo p.75
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl1.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf brc=1") << (" ")
                          << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qo_symbols_a[decoded_qo_symbols] = v;
            (decoded_qo_symbols)++;
          }
        }
        break;
      }
      break;
    }
    case 2: {
      {

        if ((thidx) <= (5)) {
          // reconstruction law block=qo thidx-choice=simple brc=2
          for (int i = 0;
               (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc2(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qo p.75
            auto v = (0.f);
            try {
              if ((mcode) < (6)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (6)) {
                  v = ((symbol_sign) * (table_b2.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("exception simple brc=2")
                          << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qo_symbols_a[decoded_qo_symbols] = v;
            (decoded_qo_symbols)++;
          }
        } else {
          // reconstruction law block=qo thidx-choice=normal brc=2
          for (int i = 0;
               (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc2(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qo p.75
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl2.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf brc=2") << (" ")
                          << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qo_symbols_a[decoded_qo_symbols] = v;
            (decoded_qo_symbols)++;
          }
        }
        break;
      }
      break;
    }
    case 3: {
      {

        if ((thidx) <= (6)) {
          // reconstruction law block=qo thidx-choice=simple brc=3
          for (int i = 0;
               (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc3(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qo p.75
            auto v = (0.f);
            try {
              if ((mcode) < (9)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (9)) {
                  v = ((symbol_sign) * (table_b3.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("exception simple brc=3")
                          << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qo_symbols_a[decoded_qo_symbols] = v;
            (decoded_qo_symbols)++;
          }
        } else {
          // reconstruction law block=qo thidx-choice=normal brc=3
          for (int i = 0;
               (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc3(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qo p.75
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl3.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf brc=3") << (" ")
                          << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qo_symbols_a[decoded_qo_symbols] = v;
            (decoded_qo_symbols)++;
          }
        }
        break;
      }
      break;
    }
    case 4: {
      {

        if ((thidx) <= (8)) {
          // reconstruction law block=qo thidx-choice=simple brc=4
          for (int i = 0;
               (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc4(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qo p.75
            auto v = (0.f);
            try {
              if ((mcode) < (15)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (15)) {
                  v = ((symbol_sign) * (table_b4.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("exception simple brc=4")
                          << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qo_symbols_a[decoded_qo_symbols] = v;
            (decoded_qo_symbols)++;
          }
        } else {
          // reconstruction law block=qo thidx-choice=normal brc=4
          for (int i = 0;
               (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
               (i)++) {
            auto sign_bit = get_sequential_bit(&s);
            auto mcode = decode_huffman_brc4(&s);
            auto symbol_sign = (1.0f);
            if (sign_bit) {
              symbol_sign = (-1.0f);
            }
            // decode qo p.75
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl4.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf brc=4") << (" ")
                          << (std::setw(8)) << (" thidx=") << (thidx)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::endl);
              assert(0);
            };
            decoded_qo_symbols_a[decoded_qo_symbols] = v;
            (decoded_qo_symbols)++;
          }
        }
        break;
      }
      break;
    }
    default: {
      {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("error brc out of range")
                    << (" ") << (std::setw(8)) << (" brc=") << (brc)
                    << (std::endl);
        assert(0);
        break;
      }
      break;
    }
    }
  }
  consume_padding_bits(&s);
  for (auto block = 0; (block) < (number_of_baq_blocks); (block) += (1)) {
    auto brc = brcs[block];
    auto thidx = thidxs[block];
    switch (brc) {
    case 0: {
      {

        // decode ie p.74 reconstruction law middle choice brc=0
        if ((thidx) <= (3)) {
          // decode ie p.74 reconstruction law simple brc=0
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_ie_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_ie_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode ie p.74 reconstruction law right side
            auto v = (0.f);
            try {
              if ((mcode) < (3)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (3)) {
                  v = ((symbol_sign) * (table_b0.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception simple block=ie brc=0") << (" ")
                          << (std::setw(8)) << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" mcode=") << (mcode) << (std::setw(8))
                          << (" packet_idx=") << (packet_idx) << (std::endl);
              assert(0);
            };
            decoded_ie_symbols_a[pos] = v;
          }
        } else {
          // decode ie p.74 reconstruction law normal brc=0
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_ie_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_ie_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode ie p.74 reconstruction law right side
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl0.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf block=ie brc=0")
                          << (" ") << (std::setw(8))
                          << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" block=") << (block) << (std::setw(8)) << (" i=")
                          << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::setw(8)) << (" pos=") << (pos)
                          << (std::setw(8)) << (" scode=") << (scode)
                          << (std::setw(8)) << (" symbol_sign=")
                          << (symbol_sign) << (std::setw(8))
                          << (" decoded_ie_symbols=") << (decoded_ie_symbols)
                          << (std::endl);
              assert(0);
            };
            decoded_ie_symbols_a[pos] = v;
          }
        }
        break;
      }
      break;
    }
    case 1: {
      {

        // decode ie p.74 reconstruction law middle choice brc=1
        if ((thidx) <= (3)) {
          // decode ie p.74 reconstruction law simple brc=1
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_ie_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_ie_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode ie p.74 reconstruction law right side
            auto v = (0.f);
            try {
              if ((mcode) < (4)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (4)) {
                  v = ((symbol_sign) * (table_b1.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception simple block=ie brc=1") << (" ")
                          << (std::setw(8)) << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" mcode=") << (mcode) << (std::setw(8))
                          << (" packet_idx=") << (packet_idx) << (std::endl);
              assert(0);
            };
            decoded_ie_symbols_a[pos] = v;
          }
        } else {
          // decode ie p.74 reconstruction law normal brc=1
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_ie_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_ie_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode ie p.74 reconstruction law right side
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl1.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf block=ie brc=1")
                          << (" ") << (std::setw(8))
                          << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" block=") << (block) << (std::setw(8)) << (" i=")
                          << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::setw(8)) << (" pos=") << (pos)
                          << (std::setw(8)) << (" scode=") << (scode)
                          << (std::setw(8)) << (" symbol_sign=")
                          << (symbol_sign) << (std::setw(8))
                          << (" decoded_ie_symbols=") << (decoded_ie_symbols)
                          << (std::endl);
              assert(0);
            };
            decoded_ie_symbols_a[pos] = v;
          }
        }
        break;
      }
      break;
    }
    case 2: {
      {

        // decode ie p.74 reconstruction law middle choice brc=2
        if ((thidx) <= (5)) {
          // decode ie p.74 reconstruction law simple brc=2
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_ie_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_ie_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode ie p.74 reconstruction law right side
            auto v = (0.f);
            try {
              if ((mcode) < (6)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (6)) {
                  v = ((symbol_sign) * (table_b2.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception simple block=ie brc=2") << (" ")
                          << (std::setw(8)) << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" mcode=") << (mcode) << (std::setw(8))
                          << (" packet_idx=") << (packet_idx) << (std::endl);
              assert(0);
            };
            decoded_ie_symbols_a[pos] = v;
          }
        } else {
          // decode ie p.74 reconstruction law normal brc=2
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_ie_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_ie_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode ie p.74 reconstruction law right side
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl2.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf block=ie brc=2")
                          << (" ") << (std::setw(8))
                          << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" block=") << (block) << (std::setw(8)) << (" i=")
                          << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::setw(8)) << (" pos=") << (pos)
                          << (std::setw(8)) << (" scode=") << (scode)
                          << (std::setw(8)) << (" symbol_sign=")
                          << (symbol_sign) << (std::setw(8))
                          << (" decoded_ie_symbols=") << (decoded_ie_symbols)
                          << (std::endl);
              assert(0);
            };
            decoded_ie_symbols_a[pos] = v;
          }
        }
        break;
      }
      break;
    }
    case 3: {
      {

        // decode ie p.74 reconstruction law middle choice brc=3
        if ((thidx) <= (6)) {
          // decode ie p.74 reconstruction law simple brc=3
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_ie_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_ie_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode ie p.74 reconstruction law right side
            auto v = (0.f);
            try {
              if ((mcode) < (9)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (9)) {
                  v = ((symbol_sign) * (table_b3.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception simple block=ie brc=3") << (" ")
                          << (std::setw(8)) << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" mcode=") << (mcode) << (std::setw(8))
                          << (" packet_idx=") << (packet_idx) << (std::endl);
              assert(0);
            };
            decoded_ie_symbols_a[pos] = v;
          }
        } else {
          // decode ie p.74 reconstruction law normal brc=3
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_ie_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_ie_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode ie p.74 reconstruction law right side
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl3.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf block=ie brc=3")
                          << (" ") << (std::setw(8))
                          << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" block=") << (block) << (std::setw(8)) << (" i=")
                          << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::setw(8)) << (" pos=") << (pos)
                          << (std::setw(8)) << (" scode=") << (scode)
                          << (std::setw(8)) << (" symbol_sign=")
                          << (symbol_sign) << (std::setw(8))
                          << (" decoded_ie_symbols=") << (decoded_ie_symbols)
                          << (std::endl);
              assert(0);
            };
            decoded_ie_symbols_a[pos] = v;
          }
        }
        break;
      }
      break;
    }
    case 4: {
      {

        // decode ie p.74 reconstruction law middle choice brc=4
        if ((thidx) <= (8)) {
          // decode ie p.74 reconstruction law simple brc=4
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_ie_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_ie_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode ie p.74 reconstruction law right side
            auto v = (0.f);
            try {
              if ((mcode) < (15)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (15)) {
                  v = ((symbol_sign) * (table_b4.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception simple block=ie brc=4") << (" ")
                          << (std::setw(8)) << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" mcode=") << (mcode) << (std::setw(8))
                          << (" packet_idx=") << (packet_idx) << (std::endl);
              assert(0);
            };
            decoded_ie_symbols_a[pos] = v;
          }
        } else {
          // decode ie p.74 reconstruction law normal brc=4
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_ie_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_ie_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode ie p.74 reconstruction law right side
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl4.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf block=ie brc=4")
                          << (" ") << (std::setw(8))
                          << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" block=") << (block) << (std::setw(8)) << (" i=")
                          << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::setw(8)) << (" pos=") << (pos)
                          << (std::setw(8)) << (" scode=") << (scode)
                          << (std::setw(8)) << (" symbol_sign=")
                          << (symbol_sign) << (std::setw(8))
                          << (" decoded_ie_symbols=") << (decoded_ie_symbols)
                          << (std::endl);
              assert(0);
            };
            decoded_ie_symbols_a[pos] = v;
          }
        }
        break;
      }
      break;
    }
    default: {
      {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("unknown brc") << (" ")
                    << (std::setw(8)) << (" static_cast<int>(brc)=")
                    << (static_cast<int>(brc)) << (std::endl);
        assert(0);
        break;
      }
      break;
    }
    }
  }
  for (auto block = 0; (block) < (number_of_baq_blocks); (block) += (1)) {
    auto brc = brcs[block];
    auto thidx = thidxs[block];
    switch (brc) {
    case 0: {
      {

        // decode io p.74 reconstruction law middle choice brc=0
        if ((thidx) <= (3)) {
          // decode io p.74 reconstruction law simple brc=0
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_io_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_io_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode io p.74 reconstruction law right side
            auto v = (0.f);
            try {
              if ((mcode) < (3)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (3)) {
                  v = ((symbol_sign) * (table_b0.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception simple block=io brc=0") << (" ")
                          << (std::setw(8)) << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" mcode=") << (mcode) << (std::setw(8))
                          << (" packet_idx=") << (packet_idx) << (std::endl);
              assert(0);
            };
            decoded_io_symbols_a[pos] = v;
          }
        } else {
          // decode io p.74 reconstruction law normal brc=0
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_io_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_io_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode io p.74 reconstruction law right side
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl0.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf block=io brc=0")
                          << (" ") << (std::setw(8))
                          << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" block=") << (block) << (std::setw(8)) << (" i=")
                          << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::setw(8)) << (" pos=") << (pos)
                          << (std::setw(8)) << (" scode=") << (scode)
                          << (std::setw(8)) << (" symbol_sign=")
                          << (symbol_sign) << (std::setw(8))
                          << (" decoded_io_symbols=") << (decoded_io_symbols)
                          << (std::endl);
              assert(0);
            };
            decoded_io_symbols_a[pos] = v;
          }
        }
        break;
      }
      break;
    }
    case 1: {
      {

        // decode io p.74 reconstruction law middle choice brc=1
        if ((thidx) <= (3)) {
          // decode io p.74 reconstruction law simple brc=1
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_io_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_io_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode io p.74 reconstruction law right side
            auto v = (0.f);
            try {
              if ((mcode) < (4)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (4)) {
                  v = ((symbol_sign) * (table_b1.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception simple block=io brc=1") << (" ")
                          << (std::setw(8)) << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" mcode=") << (mcode) << (std::setw(8))
                          << (" packet_idx=") << (packet_idx) << (std::endl);
              assert(0);
            };
            decoded_io_symbols_a[pos] = v;
          }
        } else {
          // decode io p.74 reconstruction law normal brc=1
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_io_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_io_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode io p.74 reconstruction law right side
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl1.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf block=io brc=1")
                          << (" ") << (std::setw(8))
                          << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" block=") << (block) << (std::setw(8)) << (" i=")
                          << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::setw(8)) << (" pos=") << (pos)
                          << (std::setw(8)) << (" scode=") << (scode)
                          << (std::setw(8)) << (" symbol_sign=")
                          << (symbol_sign) << (std::setw(8))
                          << (" decoded_io_symbols=") << (decoded_io_symbols)
                          << (std::endl);
              assert(0);
            };
            decoded_io_symbols_a[pos] = v;
          }
        }
        break;
      }
      break;
    }
    case 2: {
      {

        // decode io p.74 reconstruction law middle choice brc=2
        if ((thidx) <= (5)) {
          // decode io p.74 reconstruction law simple brc=2
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_io_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_io_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode io p.74 reconstruction law right side
            auto v = (0.f);
            try {
              if ((mcode) < (6)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (6)) {
                  v = ((symbol_sign) * (table_b2.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception simple block=io brc=2") << (" ")
                          << (std::setw(8)) << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" mcode=") << (mcode) << (std::setw(8))
                          << (" packet_idx=") << (packet_idx) << (std::endl);
              assert(0);
            };
            decoded_io_symbols_a[pos] = v;
          }
        } else {
          // decode io p.74 reconstruction law normal brc=2
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_io_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_io_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode io p.74 reconstruction law right side
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl2.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf block=io brc=2")
                          << (" ") << (std::setw(8))
                          << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" block=") << (block) << (std::setw(8)) << (" i=")
                          << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::setw(8)) << (" pos=") << (pos)
                          << (std::setw(8)) << (" scode=") << (scode)
                          << (std::setw(8)) << (" symbol_sign=")
                          << (symbol_sign) << (std::setw(8))
                          << (" decoded_io_symbols=") << (decoded_io_symbols)
                          << (std::endl);
              assert(0);
            };
            decoded_io_symbols_a[pos] = v;
          }
        }
        break;
      }
      break;
    }
    case 3: {
      {

        // decode io p.74 reconstruction law middle choice brc=3
        if ((thidx) <= (6)) {
          // decode io p.74 reconstruction law simple brc=3
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_io_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_io_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode io p.74 reconstruction law right side
            auto v = (0.f);
            try {
              if ((mcode) < (9)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (9)) {
                  v = ((symbol_sign) * (table_b3.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception simple block=io brc=3") << (" ")
                          << (std::setw(8)) << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" mcode=") << (mcode) << (std::setw(8))
                          << (" packet_idx=") << (packet_idx) << (std::endl);
              assert(0);
            };
            decoded_io_symbols_a[pos] = v;
          }
        } else {
          // decode io p.74 reconstruction law normal brc=3
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_io_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_io_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode io p.74 reconstruction law right side
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl3.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf block=io brc=3")
                          << (" ") << (std::setw(8))
                          << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" block=") << (block) << (std::setw(8)) << (" i=")
                          << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::setw(8)) << (" pos=") << (pos)
                          << (std::setw(8)) << (" scode=") << (scode)
                          << (std::setw(8)) << (" symbol_sign=")
                          << (symbol_sign) << (std::setw(8))
                          << (" decoded_io_symbols=") << (decoded_io_symbols)
                          << (std::endl);
              assert(0);
            };
            decoded_io_symbols_a[pos] = v;
          }
        }
        break;
      }
      break;
    }
    case 4: {
      {

        // decode io p.74 reconstruction law middle choice brc=4
        if ((thidx) <= (8)) {
          // decode io p.74 reconstruction law simple brc=4
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_io_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_io_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode io p.74 reconstruction law right side
            auto v = (0.f);
            try {
              if ((mcode) < (15)) {
                v = ((symbol_sign) * (mcode));
              } else {
                if ((mcode) == (15)) {
                  v = ((symbol_sign) * (table_b4.at(thidx)));
                } else {
                  std::setprecision(3);
                  (std::cout) << (std::setw(10))
                              << (((std::chrono::high_resolution_clock::now()
                                        .time_since_epoch()
                                        .count()) -
                                   (state._start_time)))
                              << (" ") << (__FILE__) << (":") << (__LINE__)
                              << (" ") << (__func__) << (" ")
                              << ("mcode too large") << (" ") << (std::setw(8))
                              << (" mcode=") << (mcode) << (std::endl);
                  assert(0);
                }
              }
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception simple block=io brc=4") << (" ")
                          << (std::setw(8)) << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" mcode=") << (mcode) << (std::setw(8))
                          << (" packet_idx=") << (packet_idx) << (std::endl);
              assert(0);
            };
            decoded_io_symbols_a[pos] = v;
          }
        } else {
          // decode io p.74 reconstruction law normal brc=4
          for (int i = 0; (((i) < (128)) && ((((i) + (((128) * (block))))) <
                                             (decoded_io_symbols)));
               (i)++) {
            auto pos = ((i) + (((128) * (block))));
            auto scode = decoded_io_symbols_a[pos];
            auto mcode = static_cast<int>(fabsf(scode));
            auto symbol_sign = copysignf((1.0f), scode);
            // decode io p.74 reconstruction law right side
            auto v = (0.f);
            try {
              v = ((symbol_sign) * (table_nrl4.at(mcode)) *
                   (table_sf.at(thidx)));
            } catch (std::out_of_range e) {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ")
                          << ("exception normal nrl or sf block=io brc=4")
                          << (" ") << (std::setw(8))
                          << (" static_cast<int>(thidx)=")
                          << (static_cast<int>(thidx)) << (std::setw(8))
                          << (" block=") << (block) << (std::setw(8)) << (" i=")
                          << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                          << (std::setw(8)) << (" pos=") << (pos)
                          << (std::setw(8)) << (" scode=") << (scode)
                          << (std::setw(8)) << (" symbol_sign=")
                          << (symbol_sign) << (std::setw(8))
                          << (" decoded_io_symbols=") << (decoded_io_symbols)
                          << (std::endl);
              assert(0);
            };
            decoded_io_symbols_a[pos] = v;
          }
        }
        break;
      }
      break;
    }
    default: {
      {
        std::setprecision(3);
        (std::cout) << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("unknown brc") << (" ")
                    << (std::setw(8)) << (" static_cast<int>(brc)=")
                    << (static_cast<int>(brc)) << (std::endl);
        assert(0);
        break;
      }
      break;
    }
    }
  }
  assert((decoded_ie_symbols) == (decoded_io_symbols));
  assert((decoded_ie_symbols) == (decoded_qe_symbols));
  assert((decoded_qo_symbols) == (decoded_qe_symbols));
  for (auto i = 0; (i) < (decoded_ie_symbols); (i) += (1)) {
    output[((2) * (i))].real(decoded_ie_symbols_a[i]);
    output[((2) * (i))].imag(decoded_qe_symbols_a[i]);
    output[((1) + (((2) * (i))))].real(decoded_io_symbols_a[i]);
    output[((1) + (((2) * (i))))].imag(decoded_qo_symbols_a[i]);
  }
  auto n = ((decoded_ie_symbols) + (decoded_io_symbols));
  return n;
}