#ifndef UTILS_H

#define UTILS_H

#include <array>
#include <iomanip>
#include <iostream>
#include <vector>

#include <complex>
enum { MAX_NUMBER_QUADS = 52378 }; // page 55
struct sequential_bit_t {
  size_t current_bit_count;
  uint8_t *data;
};
typedef struct sequential_bit_t sequential_bit_t;
struct anxillary_data_t {
  double x_axis_position;
  double y_axis_position;
  double z_axis_position;
  float x_velocity;
  float y_velocity;
  float z_velocity;
  uint16_t pod_solution_data_stamp_0;
  uint16_t pod_solution_data_stamp_1;
  uint16_t pod_solution_data_stamp_2;
  uint16_t pod_solution_data_stamp_3;
  float quaternion_0;
  float quaternion_1;
  float quaternion_2;
  float quaternion_3;
  float angular_rate_x;
  float angular_rate_y;
  float angular_rate_z;
  uint16_t gps_data_timestamp_0;
  uint16_t gps_data_timestamp_1;
  uint16_t gps_data_timestamp_2;
  uint16_t gps_data_timestamp_3;
  uint16_t pointing_status;
  uint16_t temperature_update_status;
  uint8_t tile_1_efe_h_temperature;
  uint8_t tile_1_efe_v_temperature;
  uint8_t tile_1_active_ta_temperature;
  uint8_t tile_2_efe_h_ta_temperature;
  uint8_t tile_2_efe_h_temperature;
  uint8_t tile_2_efe_v_temperature;
  uint8_t tile_2_active_ta_temperature;
  uint8_t tile_3_efe_h_ta_temperature;
  uint8_t tile_3_efe_h_temperature;
  uint8_t tile_3_efe_v_temperature;
  uint8_t tile_3_active_ta_temperature;
  uint8_t tile_4_efe_h_ta_temperature;
  uint8_t tile_4_efe_h_temperature;
  uint8_t tile_4_efe_v_temperature;
  uint8_t tile_4_active_ta_temperature;
  uint8_t tile_5_efe_h_ta_temperature;
  uint8_t tile_5_efe_h_temperature;
  uint8_t tile_5_efe_v_temperature;
  uint8_t tile_5_active_ta_temperature;
  uint8_t tile_6_efe_h_ta_temperature;
  uint8_t tile_6_efe_h_temperature;
  uint8_t tile_6_efe_v_temperature;
  uint8_t tile_6_active_ta_temperature;
  uint8_t tile_7_efe_h_ta_temperature;
  uint8_t tile_7_efe_h_temperature;
  uint8_t tile_7_efe_v_temperature;
  uint8_t tile_7_active_ta_temperature;
  uint8_t tile_8_efe_h_ta_temperature;
  uint8_t tile_8_efe_h_temperature;
  uint8_t tile_8_efe_v_temperature;
  uint8_t tile_8_active_ta_temperature;
  uint8_t tile_9_efe_h_ta_temperature;
  uint8_t tile_9_efe_h_temperature;
  uint8_t tile_9_efe_v_temperature;
  uint8_t tile_9_active_ta_temperature;
  uint8_t tile_10_efe_h_ta_temperature;
  uint8_t tile_10_efe_h_temperature;
  uint8_t tile_10_efe_v_temperature;
  uint8_t tile_10_active_ta_temperature;
  uint8_t tile_11_efe_h_ta_temperature;
  uint8_t tile_11_efe_h_temperature;
  uint8_t tile_11_efe_v_temperature;
  uint8_t tile_11_active_ta_temperature;
  uint8_t tile_12_efe_h_ta_temperature;
  uint8_t tile_12_efe_h_temperature;
  uint8_t tile_12_efe_v_temperature;
  uint8_t tile_12_active_ta_temperature;
  uint8_t tile_13_efe_h_ta_temperature;
  uint8_t tile_13_efe_h_temperature;
  uint8_t tile_13_efe_v_temperature;
  uint8_t tile_13_active_ta_temperature;
  uint8_t tile_14_efe_h_ta_temperature;
  uint8_t tile_14_efe_h_temperature;
  uint8_t tile_14_efe_v_temperature;
  uint8_t tile_14_active_ta_temperature;
  uint8_t tile_15_efe_h_ta_temperature;
  uint16_t tgu_temperature;
};
typedef struct anxillary_data_t anxillary_data_t;
;

#endif
