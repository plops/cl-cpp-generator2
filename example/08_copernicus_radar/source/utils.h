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
};
typedef struct anxillary_data_t anxillary_data_t;
;

#endif
