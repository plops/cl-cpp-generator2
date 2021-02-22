
#include "utils.h"

#include "globals.h"

#include "proto2.h"

extern State state;
#include <cassert>
#include <cmath>
extern const std::array<const float, 256> table_sf;
// table 5.2-1 simple reconstruction parameter values A
const std::array<const float, 4> table_a3 = {3, 3, (3.120f), (3.550f)};
const std::array<const float, 6> table_a4 = {7,        7,       7,
                                             (7.170f), (7.40f), (7.760f)};
const std::array<const float, 11> table_a5 = {
    15,        15,        15,        15,        15,       15,
    (15.440f), (15.560f), (16.110f), (16.380f), (16.650f)};
// table 5.2-2 normalized reconstruction levels
const std::array<const float, 4> table_nrla3 = {(0.2490f), (0.76810f),
                                                (1.36550f), (2.18640f)};
const std::array<const float, 8> table_nrla4 = {
    (0.1290f),  (0.390f),   (0.66010f), (0.94710f),
    (1.26230f), (1.62610f), (2.07930f), (2.74670f)};
const std::array<const float, 16> table_nrla5 = {
    (6.60e-2f), (0.19850f), (0.3320f),  (0.46770f), (0.60610f), (0.74870f),
    (0.89640f), (1.0510f),  (1.21430f), (1.38960f), (1.580f),   (1.79140f),
    (2.03290f), (2.32340f), (2.69710f), (3.26920f)};
inline int get_baq3_code(sequential_bit_t *s) {
  return ((((0x4) * (get_sequential_bit(s)))) +
          (((0x2) * (get_sequential_bit(s)))) +
          (((0x1) * (get_sequential_bit(s)))));
}
inline int get_baq4_code(sequential_bit_t *s) {
  return ((((0x8) * (get_sequential_bit(s)))) +
          (((0x4) * (get_sequential_bit(s)))) +
          (((0x2) * (get_sequential_bit(s)))) +
          (((0x1) * (get_sequential_bit(s)))));
}
inline int get_baq5_code(sequential_bit_t *s) {
  return ((((0x10) * (get_sequential_bit(s)))) +
          (((0x8) * (get_sequential_bit(s)))) +
          (((0x4) * (get_sequential_bit(s)))) +
          (((0x2) * (get_sequential_bit(s)))) +
          (((0x1) * (get_sequential_bit(s)))));
}
int init_decode_type_c_packet_baq3(int packet_idx,
                                   std::complex<float> *output) {
  auto header = state._header_data[packet_idx].data();
  auto offset = state._header_offset[packet_idx];
  auto number_of_quads =
      ((((0x1) * (header[66]))) + (((0x100) * (((0xFF) & (header[65]))))));
  auto baq_block_length = ((8) * (((1) + (((0xFF) & ((header[38]) >> (0)))))));
  auto number_of_baq_blocks =
      static_cast<int>(round(ceil((((((2.0f)) * (number_of_quads))) / (256)))));
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
    // nothing for ie
    {

      // reconstruction law block=ie thidx-choice=thidx-unknown
      for (int i = 0;
           (((i) < (128)) && ((decoded_ie_symbols) < (number_of_quads)));
           (i)++) {
        auto smcode = get_baq3_code(&s);
        auto sign_bit = ((1) & ((smcode) >> (((3) - (1)))));
        auto mcode = ((smcode) & (0x3));
        auto symbol_sign = (1.0f);
        if (sign_bit) {
          symbol_sign = (-1.0f);
        }
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later
        decoded_ie_symbols_a[decoded_ie_symbols] = v;
        (decoded_ie_symbols)++;
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
    // nothing for io
    {

      // reconstruction law block=io thidx-choice=thidx-unknown
      for (int i = 0;
           (((i) < (128)) && ((decoded_io_symbols) < (number_of_quads)));
           (i)++) {
        auto smcode = get_baq3_code(&s);
        auto sign_bit = ((1) & ((smcode) >> (((3) - (1)))));
        auto mcode = ((smcode) & (0x3));
        auto symbol_sign = (1.0f);
        if (sign_bit) {
          symbol_sign = (-1.0f);
        }
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later
        decoded_io_symbols_a[decoded_io_symbols] = v;
        (decoded_io_symbols)++;
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
    thidxs[block] = thidx;
    {

      if ((thidx) <= (3)) {
        // reconstruction law block=qe thidx-choice=simple
        for (int i = 0;
             (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq3_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((3) - (1)))));
          auto mcode = ((smcode) & (0x3));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qe p.66
          auto v = (0.f);
          try {
            if ((mcode) < (3)) {
              v = ((symbol_sign) * (mcode));
            } else {
              if ((mcode) == (3)) {
                v = ((symbol_sign) * (table_a3.at(thidx)));
              } else {
                std::setprecision(3);
                (std::cout)
                    << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("mcode too large") << (" ")
                    << (std::setw(8)) << (" mcode=") << (mcode) << (std::endl);
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
                        << (__func__) << (" ") << ("exception simple a=3")
                        << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        }
      } else {
        // reconstruction law block=qe thidx-choice=normal
        for (int i = 0;
             (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq3_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((3) - (1)))));
          auto mcode = ((smcode) & (0x3));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qe p.66
          auto v = (0.f);
          try {
            v = ((symbol_sign) * (table_nrla3.at(mcode)) *
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
                        << ("exception normal nrl or sf ") << (" ")
                        << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        }
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
    auto thidx = thidxs[block];
    {

      if ((thidx) <= (3)) {
        // reconstruction law block=qo thidx-choice=simple
        for (int i = 0;
             (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq3_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((3) - (1)))));
          auto mcode = ((smcode) & (0x3));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qo p.66
          auto v = (0.f);
          try {
            if ((mcode) < (3)) {
              v = ((symbol_sign) * (mcode));
            } else {
              if ((mcode) == (3)) {
                v = ((symbol_sign) * (table_a3.at(thidx)));
              } else {
                std::setprecision(3);
                (std::cout)
                    << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("mcode too large") << (" ")
                    << (std::setw(8)) << (" mcode=") << (mcode) << (std::endl);
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
                        << (__func__) << (" ") << ("exception simple a=3")
                        << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        }
      } else {
        // reconstruction law block=qo thidx-choice=normal
        for (int i = 0;
             (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq3_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((3) - (1)))));
          auto mcode = ((smcode) & (0x3));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qo p.66
          auto v = (0.f);
          try {
            v = ((symbol_sign) * (table_nrla3.at(mcode)) *
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
                        << ("exception normal nrl or sf ") << (" ")
                        << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        }
      }
    }
  }
  consume_padding_bits(&s);
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("decode ie and io blocks") << (" ")
              << (std::setw(8)) << (" number_of_baq_blocks=")
              << (number_of_baq_blocks) << (std::endl);
  for (auto block = 0; (block) < (number_of_baq_blocks); (block) += (1)) {
    auto thidx = thidxs[block];

    // decode ie p.66 reconstruction law middle choice a=3
    if ((thidx) <= (3)) {
      // decode ie p.66 reconstruction law simple a=3
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_ie_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_ie_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode ie p.66 reconstruction law right side
        auto v = (0.f);
        try {
          if ((mcode) < (3)) {
            v = ((symbol_sign) * (mcode));
          } else {
            if ((mcode) == (3)) {
              v = ((symbol_sign) * (table_a3.at(thidx)));
            } else {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("mcode too large") << (" ")
                          << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::endl);
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
                      << ("exception simple block=ie a=3") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" mcode=") << (mcode) << (std::setw(8))
                      << (" packet_idx=") << (packet_idx) << (std::endl);
          assert(0);
        };
        decoded_ie_symbols_a[pos] = v;
      }
    } else {
      // decode ie p.66 reconstruction law normal a=3
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_ie_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_ie_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode ie p.66 reconstruction law right side
        auto v = (0.f);
        try {
          v = ((symbol_sign) * (table_nrla3.at(mcode)) * (table_sf.at(thidx)));
        } catch (std::out_of_range e) {
          std::setprecision(3);
          (std::cout) << (std::setw(10))
                      << (((std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count()) -
                           (state._start_time)))
                      << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ")
                      << ("exception normal nrl or sf block=ie a=3") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" block=") << (block) << (std::setw(8)) << (" i=")
                      << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                      << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                      << (std::setw(8)) << (" pos=") << (pos) << (std::setw(8))
                      << (" scode=") << (scode) << (std::setw(8))
                      << (" symbol_sign=") << (symbol_sign) << (std::setw(8))
                      << (" decoded_ie_symbols=") << (decoded_ie_symbols)
                      << (std::endl);
          assert(0);
        };
        decoded_ie_symbols_a[pos] = v;
      }
    }
  }
  for (auto block = 0; (block) < (number_of_baq_blocks); (block) += (1)) {
    auto thidx = thidxs[block];

    // decode io p.66 reconstruction law middle choice a=3
    if ((thidx) <= (3)) {
      // decode io p.66 reconstruction law simple a=3
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_io_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_io_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode io p.66 reconstruction law right side
        auto v = (0.f);
        try {
          if ((mcode) < (3)) {
            v = ((symbol_sign) * (mcode));
          } else {
            if ((mcode) == (3)) {
              v = ((symbol_sign) * (table_a3.at(thidx)));
            } else {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("mcode too large") << (" ")
                          << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::endl);
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
                      << ("exception simple block=io a=3") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" mcode=") << (mcode) << (std::setw(8))
                      << (" packet_idx=") << (packet_idx) << (std::endl);
          assert(0);
        };
        decoded_io_symbols_a[pos] = v;
      }
    } else {
      // decode io p.66 reconstruction law normal a=3
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_io_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_io_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode io p.66 reconstruction law right side
        auto v = (0.f);
        try {
          v = ((symbol_sign) * (table_nrla3.at(mcode)) * (table_sf.at(thidx)));
        } catch (std::out_of_range e) {
          std::setprecision(3);
          (std::cout) << (std::setw(10))
                      << (((std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count()) -
                           (state._start_time)))
                      << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ")
                      << ("exception normal nrl or sf block=io a=3") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" block=") << (block) << (std::setw(8)) << (" i=")
                      << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                      << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                      << (std::setw(8)) << (" pos=") << (pos) << (std::setw(8))
                      << (" scode=") << (scode) << (std::setw(8))
                      << (" symbol_sign=") << (symbol_sign) << (std::setw(8))
                      << (" decoded_io_symbols=") << (decoded_io_symbols)
                      << (std::endl);
          assert(0);
        };
        decoded_io_symbols_a[pos] = v;
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
int init_decode_type_c_packet_baq4(int packet_idx,
                                   std::complex<float> *output) {
  auto header = state._header_data[packet_idx].data();
  auto offset = state._header_offset[packet_idx];
  auto number_of_quads =
      ((((0x1) * (header[66]))) + (((0x100) * (((0xFF) & (header[65]))))));
  auto baq_block_length = ((8) * (((1) + (((0xFF) & ((header[38]) >> (0)))))));
  auto number_of_baq_blocks =
      static_cast<int>(round(ceil((((((2.0f)) * (number_of_quads))) / (256)))));
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
    // nothing for ie
    {

      // reconstruction law block=ie thidx-choice=thidx-unknown
      for (int i = 0;
           (((i) < (128)) && ((decoded_ie_symbols) < (number_of_quads)));
           (i)++) {
        auto smcode = get_baq4_code(&s);
        auto sign_bit = ((1) & ((smcode) >> (((4) - (1)))));
        auto mcode = ((smcode) & (0x7));
        auto symbol_sign = (1.0f);
        if (sign_bit) {
          symbol_sign = (-1.0f);
        }
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later
        decoded_ie_symbols_a[decoded_ie_symbols] = v;
        (decoded_ie_symbols)++;
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
    // nothing for io
    {

      // reconstruction law block=io thidx-choice=thidx-unknown
      for (int i = 0;
           (((i) < (128)) && ((decoded_io_symbols) < (number_of_quads)));
           (i)++) {
        auto smcode = get_baq4_code(&s);
        auto sign_bit = ((1) & ((smcode) >> (((4) - (1)))));
        auto mcode = ((smcode) & (0x7));
        auto symbol_sign = (1.0f);
        if (sign_bit) {
          symbol_sign = (-1.0f);
        }
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later
        decoded_io_symbols_a[decoded_io_symbols] = v;
        (decoded_io_symbols)++;
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
    thidxs[block] = thidx;
    {

      if ((thidx) <= (5)) {
        // reconstruction law block=qe thidx-choice=simple
        for (int i = 0;
             (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq4_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((4) - (1)))));
          auto mcode = ((smcode) & (0x7));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qe p.66
          auto v = (0.f);
          try {
            if ((mcode) < (7)) {
              v = ((symbol_sign) * (mcode));
            } else {
              if ((mcode) == (7)) {
                v = ((symbol_sign) * (table_a4.at(thidx)));
              } else {
                std::setprecision(3);
                (std::cout)
                    << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("mcode too large") << (" ")
                    << (std::setw(8)) << (" mcode=") << (mcode) << (std::endl);
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
                        << (__func__) << (" ") << ("exception simple a=4")
                        << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        }
      } else {
        // reconstruction law block=qe thidx-choice=normal
        for (int i = 0;
             (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq4_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((4) - (1)))));
          auto mcode = ((smcode) & (0x7));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qe p.66
          auto v = (0.f);
          try {
            v = ((symbol_sign) * (table_nrla4.at(mcode)) *
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
                        << ("exception normal nrl or sf ") << (" ")
                        << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        }
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
    auto thidx = thidxs[block];
    {

      if ((thidx) <= (5)) {
        // reconstruction law block=qo thidx-choice=simple
        for (int i = 0;
             (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq4_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((4) - (1)))));
          auto mcode = ((smcode) & (0x7));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qo p.66
          auto v = (0.f);
          try {
            if ((mcode) < (7)) {
              v = ((symbol_sign) * (mcode));
            } else {
              if ((mcode) == (7)) {
                v = ((symbol_sign) * (table_a4.at(thidx)));
              } else {
                std::setprecision(3);
                (std::cout)
                    << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("mcode too large") << (" ")
                    << (std::setw(8)) << (" mcode=") << (mcode) << (std::endl);
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
                        << (__func__) << (" ") << ("exception simple a=4")
                        << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        }
      } else {
        // reconstruction law block=qo thidx-choice=normal
        for (int i = 0;
             (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq4_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((4) - (1)))));
          auto mcode = ((smcode) & (0x7));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qo p.66
          auto v = (0.f);
          try {
            v = ((symbol_sign) * (table_nrla4.at(mcode)) *
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
                        << ("exception normal nrl or sf ") << (" ")
                        << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        }
      }
    }
  }
  consume_padding_bits(&s);
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("decode ie and io blocks") << (" ")
              << (std::setw(8)) << (" number_of_baq_blocks=")
              << (number_of_baq_blocks) << (std::endl);
  for (auto block = 0; (block) < (number_of_baq_blocks); (block) += (1)) {
    auto thidx = thidxs[block];

    // decode ie p.66 reconstruction law middle choice a=4
    if ((thidx) <= (5)) {
      // decode ie p.66 reconstruction law simple a=4
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_ie_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_ie_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode ie p.66 reconstruction law right side
        auto v = (0.f);
        try {
          if ((mcode) < (7)) {
            v = ((symbol_sign) * (mcode));
          } else {
            if ((mcode) == (7)) {
              v = ((symbol_sign) * (table_a4.at(thidx)));
            } else {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("mcode too large") << (" ")
                          << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::endl);
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
                      << ("exception simple block=ie a=4") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" mcode=") << (mcode) << (std::setw(8))
                      << (" packet_idx=") << (packet_idx) << (std::endl);
          assert(0);
        };
        decoded_ie_symbols_a[pos] = v;
      }
    } else {
      // decode ie p.66 reconstruction law normal a=4
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_ie_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_ie_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode ie p.66 reconstruction law right side
        auto v = (0.f);
        try {
          v = ((symbol_sign) * (table_nrla4.at(mcode)) * (table_sf.at(thidx)));
        } catch (std::out_of_range e) {
          std::setprecision(3);
          (std::cout) << (std::setw(10))
                      << (((std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count()) -
                           (state._start_time)))
                      << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ")
                      << ("exception normal nrl or sf block=ie a=4") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" block=") << (block) << (std::setw(8)) << (" i=")
                      << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                      << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                      << (std::setw(8)) << (" pos=") << (pos) << (std::setw(8))
                      << (" scode=") << (scode) << (std::setw(8))
                      << (" symbol_sign=") << (symbol_sign) << (std::setw(8))
                      << (" decoded_ie_symbols=") << (decoded_ie_symbols)
                      << (std::endl);
          assert(0);
        };
        decoded_ie_symbols_a[pos] = v;
      }
    }
  }
  for (auto block = 0; (block) < (number_of_baq_blocks); (block) += (1)) {
    auto thidx = thidxs[block];

    // decode io p.66 reconstruction law middle choice a=4
    if ((thidx) <= (5)) {
      // decode io p.66 reconstruction law simple a=4
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_io_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_io_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode io p.66 reconstruction law right side
        auto v = (0.f);
        try {
          if ((mcode) < (7)) {
            v = ((symbol_sign) * (mcode));
          } else {
            if ((mcode) == (7)) {
              v = ((symbol_sign) * (table_a4.at(thidx)));
            } else {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("mcode too large") << (" ")
                          << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::endl);
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
                      << ("exception simple block=io a=4") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" mcode=") << (mcode) << (std::setw(8))
                      << (" packet_idx=") << (packet_idx) << (std::endl);
          assert(0);
        };
        decoded_io_symbols_a[pos] = v;
      }
    } else {
      // decode io p.66 reconstruction law normal a=4
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_io_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_io_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode io p.66 reconstruction law right side
        auto v = (0.f);
        try {
          v = ((symbol_sign) * (table_nrla4.at(mcode)) * (table_sf.at(thidx)));
        } catch (std::out_of_range e) {
          std::setprecision(3);
          (std::cout) << (std::setw(10))
                      << (((std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count()) -
                           (state._start_time)))
                      << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ")
                      << ("exception normal nrl or sf block=io a=4") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" block=") << (block) << (std::setw(8)) << (" i=")
                      << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                      << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                      << (std::setw(8)) << (" pos=") << (pos) << (std::setw(8))
                      << (" scode=") << (scode) << (std::setw(8))
                      << (" symbol_sign=") << (symbol_sign) << (std::setw(8))
                      << (" decoded_io_symbols=") << (decoded_io_symbols)
                      << (std::endl);
          assert(0);
        };
        decoded_io_symbols_a[pos] = v;
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
int init_decode_type_c_packet_baq5(int packet_idx,
                                   std::complex<float> *output) {
  auto header = state._header_data[packet_idx].data();
  auto offset = state._header_offset[packet_idx];
  auto number_of_quads =
      ((((0x1) * (header[66]))) + (((0x100) * (((0xFF) & (header[65]))))));
  auto baq_block_length = ((8) * (((1) + (((0xFF) & ((header[38]) >> (0)))))));
  auto number_of_baq_blocks =
      static_cast<int>(round(ceil((((((2.0f)) * (number_of_quads))) / (256)))));
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
    // nothing for ie
    {

      // reconstruction law block=ie thidx-choice=thidx-unknown
      for (int i = 0;
           (((i) < (128)) && ((decoded_ie_symbols) < (number_of_quads)));
           (i)++) {
        auto smcode = get_baq5_code(&s);
        auto sign_bit = ((1) & ((smcode) >> (((5) - (1)))));
        auto mcode = ((smcode) & (0xF));
        auto symbol_sign = (1.0f);
        if (sign_bit) {
          symbol_sign = (-1.0f);
        }
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later
        decoded_ie_symbols_a[decoded_ie_symbols] = v;
        (decoded_ie_symbols)++;
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
    // nothing for io
    {

      // reconstruction law block=io thidx-choice=thidx-unknown
      for (int i = 0;
           (((i) < (128)) && ((decoded_io_symbols) < (number_of_quads)));
           (i)++) {
        auto smcode = get_baq5_code(&s);
        auto sign_bit = ((1) & ((smcode) >> (((5) - (1)))));
        auto mcode = ((smcode) & (0xF));
        auto symbol_sign = (1.0f);
        if (sign_bit) {
          symbol_sign = (-1.0f);
        }
        auto v = ((symbol_sign) * (mcode));
        // in ie and io we don't have thidx yet, will be processed later
        decoded_io_symbols_a[decoded_io_symbols] = v;
        (decoded_io_symbols)++;
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
    thidxs[block] = thidx;
    {

      if ((thidx) <= (10)) {
        // reconstruction law block=qe thidx-choice=simple
        for (int i = 0;
             (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq5_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((5) - (1)))));
          auto mcode = ((smcode) & (0xF));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qe p.66
          auto v = (0.f);
          try {
            if ((mcode) < (15)) {
              v = ((symbol_sign) * (mcode));
            } else {
              if ((mcode) == (15)) {
                v = ((symbol_sign) * (table_a5.at(thidx)));
              } else {
                std::setprecision(3);
                (std::cout)
                    << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("mcode too large") << (" ")
                    << (std::setw(8)) << (" mcode=") << (mcode) << (std::endl);
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
                        << (__func__) << (" ") << ("exception simple a=5")
                        << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        }
      } else {
        // reconstruction law block=qe thidx-choice=normal
        for (int i = 0;
             (((i) < (128)) && ((decoded_qe_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq5_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((5) - (1)))));
          auto mcode = ((smcode) & (0xF));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qe p.66
          auto v = (0.f);
          try {
            v = ((symbol_sign) * (table_nrla5.at(mcode)) *
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
                        << ("exception normal nrl or sf ") << (" ")
                        << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qe_symbols_a[decoded_qe_symbols] = v;
          (decoded_qe_symbols)++;
        }
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
    auto thidx = thidxs[block];
    {

      if ((thidx) <= (10)) {
        // reconstruction law block=qo thidx-choice=simple
        for (int i = 0;
             (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq5_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((5) - (1)))));
          auto mcode = ((smcode) & (0xF));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qo p.66
          auto v = (0.f);
          try {
            if ((mcode) < (15)) {
              v = ((symbol_sign) * (mcode));
            } else {
              if ((mcode) == (15)) {
                v = ((symbol_sign) * (table_a5.at(thidx)));
              } else {
                std::setprecision(3);
                (std::cout)
                    << (std::setw(10))
                    << (((std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count()) -
                         (state._start_time)))
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__) << (" ") << ("mcode too large") << (" ")
                    << (std::setw(8)) << (" mcode=") << (mcode) << (std::endl);
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
                        << (__func__) << (" ") << ("exception simple a=5")
                        << (" ") << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        }
      } else {
        // reconstruction law block=qo thidx-choice=normal
        for (int i = 0;
             (((i) < (128)) && ((decoded_qo_symbols) < (number_of_quads)));
             (i)++) {
          auto smcode = get_baq5_code(&s);
          auto sign_bit = ((1) & ((smcode) >> (((5) - (1)))));
          auto mcode = ((smcode) & (0xF));
          auto symbol_sign = (1.0f);
          if (sign_bit) {
            symbol_sign = (-1.0f);
          }
          // decode qo p.66
          auto v = (0.f);
          try {
            v = ((symbol_sign) * (table_nrla5.at(mcode)) *
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
                        << ("exception normal nrl or sf ") << (" ")
                        << (std::setw(8)) << (" thidx=") << (thidx)
                        << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                        << (std::endl);
            assert(0);
          };
          decoded_qo_symbols_a[decoded_qo_symbols] = v;
          (decoded_qo_symbols)++;
        }
      }
    }
  }
  consume_padding_bits(&s);
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("decode ie and io blocks") << (" ")
              << (std::setw(8)) << (" number_of_baq_blocks=")
              << (number_of_baq_blocks) << (std::endl);
  for (auto block = 0; (block) < (number_of_baq_blocks); (block) += (1)) {
    auto thidx = thidxs[block];

    // decode ie p.66 reconstruction law middle choice a=5
    if ((thidx) <= (10)) {
      // decode ie p.66 reconstruction law simple a=5
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_ie_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_ie_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode ie p.66 reconstruction law right side
        auto v = (0.f);
        try {
          if ((mcode) < (15)) {
            v = ((symbol_sign) * (mcode));
          } else {
            if ((mcode) == (15)) {
              v = ((symbol_sign) * (table_a5.at(thidx)));
            } else {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("mcode too large") << (" ")
                          << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::endl);
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
                      << ("exception simple block=ie a=5") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" mcode=") << (mcode) << (std::setw(8))
                      << (" packet_idx=") << (packet_idx) << (std::endl);
          assert(0);
        };
        decoded_ie_symbols_a[pos] = v;
      }
    } else {
      // decode ie p.66 reconstruction law normal a=5
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_ie_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_ie_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode ie p.66 reconstruction law right side
        auto v = (0.f);
        try {
          v = ((symbol_sign) * (table_nrla5.at(mcode)) * (table_sf.at(thidx)));
        } catch (std::out_of_range e) {
          std::setprecision(3);
          (std::cout) << (std::setw(10))
                      << (((std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count()) -
                           (state._start_time)))
                      << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ")
                      << ("exception normal nrl or sf block=ie a=5") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" block=") << (block) << (std::setw(8)) << (" i=")
                      << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                      << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                      << (std::setw(8)) << (" pos=") << (pos) << (std::setw(8))
                      << (" scode=") << (scode) << (std::setw(8))
                      << (" symbol_sign=") << (symbol_sign) << (std::setw(8))
                      << (" decoded_ie_symbols=") << (decoded_ie_symbols)
                      << (std::endl);
          assert(0);
        };
        decoded_ie_symbols_a[pos] = v;
      }
    }
  }
  for (auto block = 0; (block) < (number_of_baq_blocks); (block) += (1)) {
    auto thidx = thidxs[block];

    // decode io p.66 reconstruction law middle choice a=5
    if ((thidx) <= (10)) {
      // decode io p.66 reconstruction law simple a=5
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_io_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_io_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode io p.66 reconstruction law right side
        auto v = (0.f);
        try {
          if ((mcode) < (15)) {
            v = ((symbol_sign) * (mcode));
          } else {
            if ((mcode) == (15)) {
              v = ((symbol_sign) * (table_a5.at(thidx)));
            } else {
              std::setprecision(3);
              (std::cout) << (std::setw(10))
                          << (((std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()) -
                               (state._start_time)))
                          << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                          << (__func__) << (" ") << ("mcode too large") << (" ")
                          << (std::setw(8)) << (" mcode=") << (mcode)
                          << (std::endl);
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
                      << ("exception simple block=io a=5") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" mcode=") << (mcode) << (std::setw(8))
                      << (" packet_idx=") << (packet_idx) << (std::endl);
          assert(0);
        };
        decoded_io_symbols_a[pos] = v;
      }
    } else {
      // decode io p.66 reconstruction law normal a=5
      for (int i = 0; (((i) < (128)) &&
                       ((((i) + (((128) * (block))))) < (decoded_io_symbols)));
           (i)++) {
        auto pos = ((i) + (((128) * (block))));
        auto scode = decoded_io_symbols_a[pos];
        auto mcode = static_cast<int>(fabsf(scode));
        auto symbol_sign = copysignf((1.0f), scode);
        // decode io p.66 reconstruction law right side
        auto v = (0.f);
        try {
          v = ((symbol_sign) * (table_nrla5.at(mcode)) * (table_sf.at(thidx)));
        } catch (std::out_of_range e) {
          std::setprecision(3);
          (std::cout) << (std::setw(10))
                      << (((std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count()) -
                           (state._start_time)))
                      << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ")
                      << ("exception normal nrl or sf block=io a=5") << (" ")
                      << (std::setw(8)) << (" static_cast<int>(thidx)=")
                      << (static_cast<int>(thidx)) << (std::setw(8))
                      << (" block=") << (block) << (std::setw(8)) << (" i=")
                      << (i) << (std::setw(8)) << (" mcode=") << (mcode)
                      << (std::setw(8)) << (" packet_idx=") << (packet_idx)
                      << (std::setw(8)) << (" pos=") << (pos) << (std::setw(8))
                      << (" scode=") << (scode) << (std::setw(8))
                      << (" symbol_sign=") << (symbol_sign) << (std::setw(8))
                      << (" decoded_io_symbols=") << (decoded_io_symbols)
                      << (std::endl);
          assert(0);
        };
        decoded_io_symbols_a[pos] = v;
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