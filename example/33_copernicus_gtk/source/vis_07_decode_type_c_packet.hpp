#ifndef VIS_07_DECODE_TYPE_C_PACKET_H
#define VIS_07_DECODE_TYPE_C_PACKET_H
#include "utils.h"
;
#include "globals.h"
;
#include <cassert>
#include <cmath>
#include <thread>
;
#include "vis_04_decode_packet.hpp"
;
inline int get_baq3_code (sequential_bit_t* s)  ;  
inline int get_baq4_code (sequential_bit_t* s)  ;  
inline int get_baq5_code (sequential_bit_t* s)  ;  
int init_decode_type_c_packet_baq3 (int packet_idx, std::complex<float>* output)  ;  
int init_decode_type_c_packet_baq4 (int packet_idx, std::complex<float>* output)  ;  
int init_decode_type_c_packet_baq5 (int packet_idx, std::complex<float>* output)  ;  
#endif