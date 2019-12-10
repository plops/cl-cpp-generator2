
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <cassert>
#include <cstring>
#include <fstream>

void init_sub_commutated_data_decoder() {
  state._ancillary_data_index = 0;
  for (int i = 0; i < state._ancillary_data_valid.size(); (i) += (1)) {
    state._ancillary_data_valid.at(i) = false;
  }
}
bool feed_sub_commutated_data_decoder(uint16_t word, int idx,
                                      int space_packet_count) {
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
    auto x_axis_position = state._ancillary_decoded.x_axis_position;
    auto y_axis_position = state._ancillary_decoded.y_axis_position;
    auto z_axis_position = state._ancillary_decoded.z_axis_position;
    auto x_velocity = state._ancillary_decoded.x_velocity;
    auto y_velocity = state._ancillary_decoded.y_velocity;
    auto z_velocity = state._ancillary_decoded.z_velocity;
    auto pod_solution_data_stamp_0 =
        state._ancillary_decoded.pod_solution_data_stamp_0;
    auto pod_solution_data_stamp_1 =
        state._ancillary_decoded.pod_solution_data_stamp_1;
    auto pod_solution_data_stamp_2 =
        state._ancillary_decoded.pod_solution_data_stamp_2;
    auto pod_solution_data_stamp_3 =
        state._ancillary_decoded.pod_solution_data_stamp_3;
    auto quaternion_0 = state._ancillary_decoded.quaternion_0;
    auto quaternion_1 = state._ancillary_decoded.quaternion_1;
    auto quaternion_2 = state._ancillary_decoded.quaternion_2;
    auto quaternion_3 = state._ancillary_decoded.quaternion_3;
    auto angular_rate_x = state._ancillary_decoded.angular_rate_x;
    auto angular_rate_y = state._ancillary_decoded.angular_rate_y;
    auto angular_rate_z = state._ancillary_decoded.angular_rate_z;
    auto gps_data_timestamp_0 = state._ancillary_decoded.gps_data_timestamp_0;
    auto gps_data_timestamp_1 = state._ancillary_decoded.gps_data_timestamp_1;
    auto gps_data_timestamp_2 = state._ancillary_decoded.gps_data_timestamp_2;
    auto gps_data_timestamp_3 = state._ancillary_decoded.gps_data_timestamp_3;
    auto pointing_status = state._ancillary_decoded.pointing_status;
    auto temperature_update_status =
        state._ancillary_decoded.temperature_update_status;
    auto tile_1_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_1_efe_h_temperature);
    auto tile_1_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_1_efe_v_temperature);
    auto tile_1_active_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_1_active_ta_temperature);
    auto tile_2_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_2_efe_h_ta_temperature);
    auto tile_2_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_2_efe_h_temperature);
    auto tile_2_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_2_efe_v_temperature);
    auto tile_2_active_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_2_active_ta_temperature);
    auto tile_3_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_3_efe_h_ta_temperature);
    auto tile_3_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_3_efe_h_temperature);
    auto tile_3_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_3_efe_v_temperature);
    auto tile_3_active_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_3_active_ta_temperature);
    auto tile_4_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_4_efe_h_ta_temperature);
    auto tile_4_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_4_efe_h_temperature);
    auto tile_4_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_4_efe_v_temperature);
    auto tile_4_active_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_4_active_ta_temperature);
    auto tile_5_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_5_efe_h_ta_temperature);
    auto tile_5_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_5_efe_h_temperature);
    auto tile_5_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_5_efe_v_temperature);
    auto tile_5_active_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_5_active_ta_temperature);
    auto tile_6_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_6_efe_h_ta_temperature);
    auto tile_6_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_6_efe_h_temperature);
    auto tile_6_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_6_efe_v_temperature);
    auto tile_6_active_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_6_active_ta_temperature);
    auto tile_7_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_7_efe_h_ta_temperature);
    auto tile_7_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_7_efe_h_temperature);
    auto tile_7_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_7_efe_v_temperature);
    auto tile_7_active_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_7_active_ta_temperature);
    auto tile_8_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_8_efe_h_ta_temperature);
    auto tile_8_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_8_efe_h_temperature);
    auto tile_8_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_8_efe_v_temperature);
    auto tile_8_active_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_8_active_ta_temperature);
    auto tile_9_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_9_efe_h_ta_temperature);
    auto tile_9_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_9_efe_h_temperature);
    auto tile_9_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_9_efe_v_temperature);
    auto tile_9_active_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_9_active_ta_temperature);
    auto tile_10_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_10_efe_h_ta_temperature);
    auto tile_10_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_10_efe_h_temperature);
    auto tile_10_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_10_efe_v_temperature);
    auto tile_10_active_ta_temperature = static_cast<int>(
        state._ancillary_decoded.tile_10_active_ta_temperature);
    auto tile_11_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_11_efe_h_ta_temperature);
    auto tile_11_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_11_efe_h_temperature);
    auto tile_11_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_11_efe_v_temperature);
    auto tile_11_active_ta_temperature = static_cast<int>(
        state._ancillary_decoded.tile_11_active_ta_temperature);
    auto tile_12_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_12_efe_h_ta_temperature);
    auto tile_12_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_12_efe_h_temperature);
    auto tile_12_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_12_efe_v_temperature);
    auto tile_12_active_ta_temperature = static_cast<int>(
        state._ancillary_decoded.tile_12_active_ta_temperature);
    auto tile_13_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_13_efe_h_ta_temperature);
    auto tile_13_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_13_efe_h_temperature);
    auto tile_13_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_13_efe_v_temperature);
    auto tile_13_active_ta_temperature = static_cast<int>(
        state._ancillary_decoded.tile_13_active_ta_temperature);
    auto tile_14_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_14_efe_h_ta_temperature);
    auto tile_14_efe_h_temperature =
        static_cast<int>(state._ancillary_decoded.tile_14_efe_h_temperature);
    auto tile_14_efe_v_temperature =
        static_cast<int>(state._ancillary_decoded.tile_14_efe_v_temperature);
    auto tile_14_active_ta_temperature = static_cast<int>(
        state._ancillary_decoded.tile_14_active_ta_temperature);
    auto tile_15_efe_h_ta_temperature =
        static_cast<int>(state._ancillary_decoded.tile_15_efe_h_ta_temperature);
    auto tgu_temperature = state._ancillary_decoded.tgu_temperature;
    {
      std::ofstream outfile;
      outfile.open("./o_anxillary.csv",
                   ((std::ios_base::out) | (std::ios_base::app)));
      if ((0) == (outfile.tellp())) {
        (outfile)
            << ("space_packet_count,x_axis_position,y_axis_position,z_axis_"
                "position,x_velocity,y_velocity,z_velocity,pod_solution_data_"
                "stamp_0,pod_solution_data_stamp_1,pod_solution_data_stamp_2,"
                "pod_solution_data_stamp_3,quaternion_0,quaternion_1,"
                "quaternion_2,quaternion_3,angular_rate_x,angular_rate_y,"
                "angular_rate_z,gps_data_timestamp_0,gps_data_timestamp_1,gps_"
                "data_timestamp_2,gps_data_timestamp_3,pointing_status,"
                "temperature_update_status,tile_1_efe_h_temperature,tile_1_efe_"
                "v_temperature,tile_1_active_ta_temperature,tile_2_efe_h_ta_"
                "temperature,tile_2_efe_h_temperature,tile_2_efe_v_temperature,"
                "tile_2_active_ta_temperature,tile_3_efe_h_ta_temperature,tile_"
                "3_efe_h_temperature,tile_3_efe_v_temperature,tile_3_active_ta_"
                "temperature,tile_4_efe_h_ta_temperature,tile_4_efe_h_"
                "temperature,tile_4_efe_v_temperature,tile_4_active_ta_"
                "temperature,tile_5_efe_h_ta_temperature,tile_5_efe_h_"
                "temperature,tile_5_efe_v_temperature,tile_5_active_ta_"
                "temperature,tile_6_efe_h_ta_temperature,tile_6_efe_h_"
                "temperature,tile_6_efe_v_temperature,tile_6_active_ta_"
                "temperature,tile_7_efe_h_ta_temperature,tile_7_efe_h_"
                "temperature,tile_7_efe_v_temperature,tile_7_active_ta_"
                "temperature,tile_8_efe_h_ta_temperature,tile_8_efe_h_"
                "temperature,tile_8_efe_v_temperature,tile_8_active_ta_"
                "temperature,tile_9_efe_h_ta_temperature,tile_9_efe_h_"
                "temperature,tile_9_efe_v_temperature,tile_9_active_ta_"
                "temperature,tile_10_efe_h_ta_temperature,tile_10_efe_h_"
                "temperature,tile_10_efe_v_temperature,tile_10_active_ta_"
                "temperature,tile_11_efe_h_ta_temperature,tile_11_efe_h_"
                "temperature,tile_11_efe_v_temperature,tile_11_active_ta_"
                "temperature,tile_12_efe_h_ta_temperature,tile_12_efe_h_"
                "temperature,tile_12_efe_v_temperature,tile_12_active_ta_"
                "temperature,tile_13_efe_h_ta_temperature,tile_13_efe_h_"
                "temperature,tile_13_efe_v_temperature,tile_13_active_ta_"
                "temperature,tile_14_efe_h_ta_temperature,tile_14_efe_h_"
                "temperature,tile_14_efe_v_temperature,tile_14_active_ta_"
                "temperature,tile_15_efe_h_ta_temperature,tgu_temperature")
            << (std::endl);
      };
      (outfile)
          << (space_packet_count) << (",") << (x_axis_position) << (",")
          << (y_axis_position) << (",") << (z_axis_position) << (",")
          << (x_velocity) << (",") << (y_velocity) << (",") << (z_velocity)
          << (",") << (pod_solution_data_stamp_0) << (",")
          << (pod_solution_data_stamp_1) << (",") << (pod_solution_data_stamp_2)
          << (",") << (pod_solution_data_stamp_3) << (",") << (quaternion_0)
          << (",") << (quaternion_1) << (",") << (quaternion_2) << (",")
          << (quaternion_3) << (",") << (angular_rate_x) << (",")
          << (angular_rate_y) << (",") << (angular_rate_z) << (",")
          << (gps_data_timestamp_0) << (",") << (gps_data_timestamp_1) << (",")
          << (gps_data_timestamp_2) << (",") << (gps_data_timestamp_3) << (",")
          << (pointing_status) << (",") << (temperature_update_status) << (",")
          << (tile_1_efe_h_temperature) << (",") << (tile_1_efe_v_temperature)
          << (",") << (tile_1_active_ta_temperature) << (",")
          << (tile_2_efe_h_ta_temperature) << (",")
          << (tile_2_efe_h_temperature) << (",") << (tile_2_efe_v_temperature)
          << (",") << (tile_2_active_ta_temperature) << (",")
          << (tile_3_efe_h_ta_temperature) << (",")
          << (tile_3_efe_h_temperature) << (",") << (tile_3_efe_v_temperature)
          << (",") << (tile_3_active_ta_temperature) << (",")
          << (tile_4_efe_h_ta_temperature) << (",")
          << (tile_4_efe_h_temperature) << (",") << (tile_4_efe_v_temperature)
          << (",") << (tile_4_active_ta_temperature) << (",")
          << (tile_5_efe_h_ta_temperature) << (",")
          << (tile_5_efe_h_temperature) << (",") << (tile_5_efe_v_temperature)
          << (",") << (tile_5_active_ta_temperature) << (",")
          << (tile_6_efe_h_ta_temperature) << (",")
          << (tile_6_efe_h_temperature) << (",") << (tile_6_efe_v_temperature)
          << (",") << (tile_6_active_ta_temperature) << (",")
          << (tile_7_efe_h_ta_temperature) << (",")
          << (tile_7_efe_h_temperature) << (",") << (tile_7_efe_v_temperature)
          << (",") << (tile_7_active_ta_temperature) << (",")
          << (tile_8_efe_h_ta_temperature) << (",")
          << (tile_8_efe_h_temperature) << (",") << (tile_8_efe_v_temperature)
          << (",") << (tile_8_active_ta_temperature) << (",")
          << (tile_9_efe_h_ta_temperature) << (",")
          << (tile_9_efe_h_temperature) << (",") << (tile_9_efe_v_temperature)
          << (",") << (tile_9_active_ta_temperature) << (",")
          << (tile_10_efe_h_ta_temperature) << (",")
          << (tile_10_efe_h_temperature) << (",") << (tile_10_efe_v_temperature)
          << (",") << (tile_10_active_ta_temperature) << (",")
          << (tile_11_efe_h_ta_temperature) << (",")
          << (tile_11_efe_h_temperature) << (",") << (tile_11_efe_v_temperature)
          << (",") << (tile_11_active_ta_temperature) << (",")
          << (tile_12_efe_h_ta_temperature) << (",")
          << (tile_12_efe_h_temperature) << (",") << (tile_12_efe_v_temperature)
          << (",") << (tile_12_active_ta_temperature) << (",")
          << (tile_13_efe_h_ta_temperature) << (",")
          << (tile_13_efe_h_temperature) << (",") << (tile_13_efe_v_temperature)
          << (",") << (tile_13_active_ta_temperature) << (",")
          << (tile_14_efe_h_ta_temperature) << (",")
          << (tile_14_efe_h_temperature) << (",") << (tile_14_efe_v_temperature)
          << (",") << (tile_14_active_ta_temperature) << (",")
          << (tile_15_efe_h_ta_temperature) << (",") << (tgu_temperature)
          << (std::endl);
      outfile.close();
    };
    return true;
  } else {
    return false;
  }
};