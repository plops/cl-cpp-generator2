
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <cassert>
#include <cstring>

void init_sub_commutated_data_decoder() {
  state._ancillary_data_index = 0;
  for (int i = 0; i < state._ancillary_data_valid.size(); (i) += (1)) {
    state._ancillary_data_valid.at(i) = false;
  }
}
bool feed_sub_commutated_data_decoder(uint16_t word, int idx) {
  state._ancillary_data_index = idx;
  state._ancillary_data.at(state._ancillary_data_index) = word;
  state._ancillary_data_valid.at(state._ancillary_data_index) = true;
  if ((state._ancillary_data_index) ==
      (((state._ancillary_data.size()) - (1)))) {
    assert(state._ancillary_data_valid.at(1));
    assert(state._ancillary_data_valid.at(2));
    assert(state._ancillary_data_valid.at(3));
    assert(state._ancillary_data_valid.at(4));
    assert(state._ancillary_data_valid.at(5));
    assert(state._ancillary_data_valid.at(6));
    assert(state._ancillary_data_valid.at(7));
    assert(state._ancillary_data_valid.at(8));
    assert(state._ancillary_data_valid.at(9));
    assert(state._ancillary_data_valid.at(10));
    assert(state._ancillary_data_valid.at(11));
    assert(state._ancillary_data_valid.at(12));
    assert(state._ancillary_data_valid.at(13));
    assert(state._ancillary_data_valid.at(14));
    assert(state._ancillary_data_valid.at(15));
    assert(state._ancillary_data_valid.at(16));
    assert(state._ancillary_data_valid.at(17));
    assert(state._ancillary_data_valid.at(18));
    assert(state._ancillary_data_valid.at(19));
    assert(state._ancillary_data_valid.at(20));
    assert(state._ancillary_data_valid.at(21));
    assert(state._ancillary_data_valid.at(22));
    assert(state._ancillary_data_valid.at(23));
    assert(state._ancillary_data_valid.at(24));
    assert(state._ancillary_data_valid.at(25));
    assert(state._ancillary_data_valid.at(26));
    assert(state._ancillary_data_valid.at(27));
    assert(state._ancillary_data_valid.at(28));
    assert(state._ancillary_data_valid.at(29));
    assert(state._ancillary_data_valid.at(30));
    assert(state._ancillary_data_valid.at(31));
    assert(state._ancillary_data_valid.at(32));
    assert(state._ancillary_data_valid.at(33));
    assert(state._ancillary_data_valid.at(34));
    assert(state._ancillary_data_valid.at(35));
    assert(state._ancillary_data_valid.at(36));
    assert(state._ancillary_data_valid.at(37));
    assert(state._ancillary_data_valid.at(38));
    assert(state._ancillary_data_valid.at(39));
    assert(state._ancillary_data_valid.at(40));
    assert(state._ancillary_data_valid.at(41));
    assert(state._ancillary_data_valid.at(42));
    assert(state._ancillary_data_valid.at(43));
    assert(state._ancillary_data_valid.at(44));
    assert(state._ancillary_data_valid.at(45));
    assert(state._ancillary_data_valid.at(46));
    assert(state._ancillary_data_valid.at(47));
    assert(state._ancillary_data_valid.at(48));
    assert(state._ancillary_data_valid.at(49));
    assert(state._ancillary_data_valid.at(50));
    assert(state._ancillary_data_valid.at(51));
    assert(state._ancillary_data_valid.at(52));
    assert(state._ancillary_data_valid.at(53));
    assert(state._ancillary_data_valid.at(54));
    assert(state._ancillary_data_valid.at(55));
    assert(state._ancillary_data_valid.at(56));
    assert(state._ancillary_data_valid.at(57));
    assert(state._ancillary_data_valid.at(58));
    assert(state._ancillary_data_valid.at(59));
    assert(state._ancillary_data_valid.at(60));
    assert(state._ancillary_data_valid.at(61));
    assert(state._ancillary_data_valid.at(62));
    assert(state._ancillary_data_valid.at(63));
    assert(state._ancillary_data_valid.at(64));
    memcpy(reinterpret_cast<void *>(&(state._ancillary_decoded)),
           reinterpret_cast<void *>(state._ancillary_data.data()),
           sizeof(state._ancillary_data));
    init_sub_commutated_data_decoder();
    (std::cout)
        << (std::setw(10))
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (state._start_time)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (std::endl) << ("    x_axis_position=")
        << (state._ancillary_decoded.x_axis_position) << (std::endl)
        << ("    y_axis_position=")
        << (state._ancillary_decoded.y_axis_position) << (std::endl)
        << ("    z_axis_position=")
        << (state._ancillary_decoded.z_axis_position) << (std::endl)
        << ("    x_velocity=") << (state._ancillary_decoded.x_velocity)
        << (std::endl) << ("    y_velocity=")
        << (state._ancillary_decoded.y_velocity) << (std::endl)
        << ("    z_velocity=") << (state._ancillary_decoded.z_velocity)
        << (std::endl) << ("    pod_solution_data_stamp_0=")
        << (state._ancillary_decoded.pod_solution_data_stamp_0) << (std::endl)
        << ("    pod_solution_data_stamp_1=")
        << (state._ancillary_decoded.pod_solution_data_stamp_1) << (std::endl)
        << ("    pod_solution_data_stamp_2=")
        << (state._ancillary_decoded.pod_solution_data_stamp_2) << (std::endl)
        << ("    pod_solution_data_stamp_3=")
        << (state._ancillary_decoded.pod_solution_data_stamp_3) << (std::endl)
        << ("    quaternion_0=") << (state._ancillary_decoded.quaternion_0)
        << (std::endl) << ("    quaternion_1=")
        << (state._ancillary_decoded.quaternion_1) << (std::endl)
        << ("    quaternion_2=") << (state._ancillary_decoded.quaternion_2)
        << (std::endl) << ("    quaternion_3=")
        << (state._ancillary_decoded.quaternion_3) << (std::endl)
        << ("    angular_rate_x=") << (state._ancillary_decoded.angular_rate_x)
        << (std::endl) << ("    angular_rate_y=")
        << (state._ancillary_decoded.angular_rate_y) << (std::endl)
        << ("    angular_rate_z=") << (state._ancillary_decoded.angular_rate_z)
        << (std::endl) << ("    gps_data_timestamp_0=")
        << (state._ancillary_decoded.gps_data_timestamp_0) << (std::endl)
        << ("    gps_data_timestamp_1=")
        << (state._ancillary_decoded.gps_data_timestamp_1) << (std::endl)
        << ("    gps_data_timestamp_2=")
        << (state._ancillary_decoded.gps_data_timestamp_2) << (std::endl)
        << ("    gps_data_timestamp_3=")
        << (state._ancillary_decoded.gps_data_timestamp_3) << (std::endl)
        << ("    pointing_status=")
        << (state._ancillary_decoded.pointing_status) << (std::endl)
        << ("    temperature_update_status=")
        << (state._ancillary_decoded.temperature_update_status) << (std::endl)
        << ("    tile_1_efe_h_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_1_efe_h_temperature))
        << (std::endl) << ("    tile_1_efe_v_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_1_efe_v_temperature))
        << (std::endl) << ("    tile_1_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_1_active_ta_temperature))
        << (std::endl) << ("    tile_2_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_2_efe_h_ta_temperature))
        << (std::endl) << ("    tile_2_efe_h_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_2_efe_h_temperature))
        << (std::endl) << ("    tile_2_efe_v_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_2_efe_v_temperature))
        << (std::endl) << ("    tile_2_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_2_active_ta_temperature))
        << (std::endl) << ("    tile_3_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_3_efe_h_ta_temperature))
        << (std::endl) << ("    tile_3_efe_h_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_3_efe_h_temperature))
        << (std::endl) << ("    tile_3_efe_v_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_3_efe_v_temperature))
        << (std::endl) << ("    tile_3_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_3_active_ta_temperature))
        << (std::endl) << ("    tile_4_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_4_efe_h_ta_temperature))
        << (std::endl) << ("    tile_4_efe_h_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_4_efe_h_temperature))
        << (std::endl) << ("    tile_4_efe_v_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_4_efe_v_temperature))
        << (std::endl) << ("    tile_4_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_4_active_ta_temperature))
        << (std::endl) << ("    tile_5_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_5_efe_h_ta_temperature))
        << (std::endl) << ("    tile_5_efe_h_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_5_efe_h_temperature))
        << (std::endl) << ("    tile_5_efe_v_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_5_efe_v_temperature))
        << (std::endl) << ("    tile_5_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_5_active_ta_temperature))
        << (std::endl) << ("    tile_6_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_6_efe_h_ta_temperature))
        << (std::endl) << ("    tile_6_efe_h_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_6_efe_h_temperature))
        << (std::endl) << ("    tile_6_efe_v_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_6_efe_v_temperature))
        << (std::endl) << ("    tile_6_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_6_active_ta_temperature))
        << (std::endl) << ("    tile_7_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_7_efe_h_ta_temperature))
        << (std::endl) << ("    tile_7_efe_h_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_7_efe_h_temperature))
        << (std::endl) << ("    tile_7_efe_v_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_7_efe_v_temperature))
        << (std::endl) << ("    tile_7_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_7_active_ta_temperature))
        << (std::endl) << ("    tile_8_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_8_efe_h_ta_temperature))
        << (std::endl) << ("    tile_8_efe_h_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_8_efe_h_temperature))
        << (std::endl) << ("    tile_8_efe_v_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_8_efe_v_temperature))
        << (std::endl) << ("    tile_8_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_8_active_ta_temperature))
        << (std::endl) << ("    tile_9_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_9_efe_h_ta_temperature))
        << (std::endl) << ("    tile_9_efe_h_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_9_efe_h_temperature))
        << (std::endl) << ("    tile_9_efe_v_temperature=")
        << (static_cast<int>(state._ancillary_decoded.tile_9_efe_v_temperature))
        << (std::endl) << ("    tile_9_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_9_active_ta_temperature))
        << (std::endl) << ("    tile_10_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_10_efe_h_ta_temperature))
        << (std::endl) << ("    tile_10_efe_h_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_10_efe_h_temperature))
        << (std::endl) << ("    tile_10_efe_v_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_10_efe_v_temperature))
        << (std::endl) << ("    tile_10_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_10_active_ta_temperature))
        << (std::endl) << ("    tile_11_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_11_efe_h_ta_temperature))
        << (std::endl) << ("    tile_11_efe_h_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_11_efe_h_temperature))
        << (std::endl) << ("    tile_11_efe_v_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_11_efe_v_temperature))
        << (std::endl) << ("    tile_11_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_11_active_ta_temperature))
        << (std::endl) << ("    tile_12_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_12_efe_h_ta_temperature))
        << (std::endl) << ("    tile_12_efe_h_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_12_efe_h_temperature))
        << (std::endl) << ("    tile_12_efe_v_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_12_efe_v_temperature))
        << (std::endl) << ("    tile_12_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_12_active_ta_temperature))
        << (std::endl) << ("    tile_13_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_13_efe_h_ta_temperature))
        << (std::endl) << ("    tile_13_efe_h_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_13_efe_h_temperature))
        << (std::endl) << ("    tile_13_efe_v_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_13_efe_v_temperature))
        << (std::endl) << ("    tile_13_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_13_active_ta_temperature))
        << (std::endl) << ("    tile_14_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_14_efe_h_ta_temperature))
        << (std::endl) << ("    tile_14_efe_h_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_14_efe_h_temperature))
        << (std::endl) << ("    tile_14_efe_v_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_14_efe_v_temperature))
        << (std::endl) << ("    tile_14_active_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_14_active_ta_temperature))
        << (std::endl) << ("    tile_15_efe_h_ta_temperature=")
        << (static_cast<int>(
               state._ancillary_decoded.tile_15_efe_h_ta_temperature))
        << (std::endl) << ("    tgu_temperature=")
        << (state._ancillary_decoded.tgu_temperature) << (std::endl);
    return true;
  } else {
    return false;
  }
};