
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>
#include <unordered_map>

// implementation
#include "vis_00_base.hpp"
#include "vis_01_mmap.hpp"
#include "vis_02_collect_packet_headers.hpp"
#include "vis_06_decode_sub_commutated_data.hpp"

State state = {};
CellItem_SpacePacketHeader0::CellItem_SpacePacketHeader0() : m_offset(0) {}
CellItem_SpacePacketHeader0::~CellItem_SpacePacketHeader0() {}
CellItem_SpacePacketHeader0::CellItem_SpacePacketHeader0(
    const CellItem_SpacePacketHeader0 &src) {
  operator=(src);
}
CellItem_SpacePacketHeader0::CellItem_SpacePacketHeader0(gsize offset)
    : m_offset(offset) {
  auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
  m_packet_version_number = static_cast<int>(((0x7) & ((p[0]) >> (5))));
  m_packet_type = static_cast<int>(((0x1) & ((p[0]) >> (4))));
  m_secondary_header_flag = static_cast<int>(((0x1) & ((p[0]) >> (3))));
  m_application_process_id_process_id = static_cast<int>(
      (((((0xF0) & (p[1]))) >> (4)) + (((0x10) * (((0x7) & (p[0])))))));
  m_application_process_id_packet_category =
      static_cast<int>(((0xF) & ((p[1]) >> (0))));
  m_sequence_flags = static_cast<int>(((0x3) & ((p[2]) >> (6))));
  m_sequence_count = static_cast<int>(
      ((((0x1) * (p[3]))) + (((0x100) * (((0x3F) & (p[2])))))));
  m_data_length = static_cast<int>(
      ((((0x1) * (p[5]))) + (((0x100) * (((0xFF) & (p[4])))))));
  m_coarse_time = static_cast<int>(((((0x1) * (p[9]))) + (((0x100) * (p[8]))) +
                                    (((0x10000) * (p[7]))) +
                                    (((0x1000000) * (((0xFF) & (p[6])))))));
  m_fine_time = static_cast<int>(
      ((((0x1) * (p[11]))) + (((0x100) * (((0xFF) & (p[10])))))));
  m_sync_marker = static_cast<int>(
      ((((0x1) * (p[15]))) + (((0x100) * (p[14]))) + (((0x10000) * (p[13]))) +
       (((0x1000000) * (((0xFF) & (p[12])))))));
  m_data_take_id = static_cast<int>(
      ((((0x1) * (p[19]))) + (((0x100) * (p[18]))) + (((0x10000) * (p[17]))) +
       (((0x1000000) * (((0xFF) & (p[16])))))));
  m_ecc_number = static_cast<int>(((0xFF) & ((p[20]) >> (0))));
  m_ignore_0 = static_cast<int>(((0x1) & ((p[21]) >> (7))));
  m_test_mode = static_cast<int>(((0x7) & ((p[21]) >> (4))));
  m_rx_channel_id = static_cast<int>(((0xF) & ((p[21]) >> (0))));
  m_instrument_configuration_id = static_cast<int>(
      ((((0x1) * (p[25]))) + (((0x100) * (p[24]))) + (((0x10000) * (p[23]))) +
       (((0x1000000) * (((0xFF) & (p[22])))))));
  m_sub_commutated_index = static_cast<int>(((0xFF) & ((p[26]) >> (0))));
  m_sub_commutated_data = static_cast<int>(
      ((((0x1) * (p[28]))) + (((0x100) * (((0xFF) & (p[27])))))));
  m_space_packet_count = static_cast<int>(
      ((((0x1) * (p[32]))) + (((0x100) * (p[31]))) + (((0x10000) * (p[30]))) +
       (((0x1000000) * (((0xFF) & (p[29])))))));
  m_pri_count = static_cast<int>(((((0x1) * (p[36]))) + (((0x100) * (p[35]))) +
                                  (((0x10000) * (p[34]))) +
                                  (((0x1000000) * (((0xFF) & (p[33])))))));
  m_error_flag = static_cast<int>(((0x1) & ((p[37]) >> (7))));
  m_ignore_1 = static_cast<int>(((0x3) & ((p[37]) >> (5))));
  m_baq_mode = static_cast<int>(((0x1F) & ((p[37]) >> (0))));
  m_baq_block_length = static_cast<int>(((0xFF) & ((p[38]) >> (0))));
  m_ignore_2 = static_cast<int>(((0xFF) & ((p[39]) >> (0))));
  m_range_decimation = static_cast<int>(((0xFF) & ((p[40]) >> (0))));
  m_rx_gain = static_cast<int>(((0xFF) & ((p[41]) >> (0))));
  m_tx_ramp_rate_polarity = static_cast<int>(((0x1) & ((p[42]) >> (7))));
  m_tx_ramp_rate_magnitude = static_cast<int>(
      ((((0x1) * (p[43]))) + (((0x100) * (((0x7F) & (p[42])))))));
  m_tx_pulse_start_frequency_polarity =
      static_cast<int>(((0x1) & ((p[44]) >> (7))));
  m_tx_pulse_start_frequency_magnitude = static_cast<int>(
      ((((0x1) * (p[45]))) + (((0x100) * (((0x7F) & (p[44])))))));
  m_tx_pulse_length =
      static_cast<int>(((((0x1) * (p[48]))) + (((0x100) * (p[47]))) +
                        (((0x10000) * (((0xFF) & (p[46])))))));
  m_ignore_3 = static_cast<int>(((0x7) & ((p[49]) >> (5))));
  m_rank = static_cast<int>(((0x1F) & ((p[49]) >> (0))));
  m_pulse_repetition_interval =
      static_cast<int>(((((0x1) * (p[52]))) + (((0x100) * (p[51]))) +
                        (((0x10000) * (((0xFF) & (p[50])))))));
  m_sampling_window_start_time =
      static_cast<int>(((((0x1) * (p[55]))) + (((0x100) * (p[54]))) +
                        (((0x10000) * (((0xFF) & (p[53])))))));
  m_sampling_window_length =
      static_cast<int>(((((0x1) * (p[58]))) + (((0x100) * (p[57]))) +
                        (((0x10000) * (((0xFF) & (p[56])))))));
  m_sab_ssb_calibration_p = static_cast<int>(((0x1) & ((p[59]) >> (7))));
  m_sab_ssb_polarisation = static_cast<int>(((0x7) & ((p[59]) >> (4))));
  m_sab_ssb_temp_comp = static_cast<int>(((0x3) & ((p[59]) >> (2))));
  m_sab_ssb_ignore_0 = static_cast<int>(((0x3) & ((p[59]) >> (0))));
  m_sab_ssb_elevation_beam_address =
      static_cast<int>(((0xF) & ((p[60]) >> (4))));
  m_sab_ssb_ignore_1 = static_cast<int>(((0x3) & ((p[60]) >> (2))));
  m_sab_ssb_azimuth_beam_address = static_cast<int>(
      ((((0x1) * (p[61]))) + (((0x100) * (((0x3) & (p[60])))))));
  m_ses_ssb_cal_mode = static_cast<int>(((0x3) & ((p[62]) >> (6))));
  m_ses_ssb_ignore_0 = static_cast<int>(((0x1) & ((p[62]) >> (5))));
  m_ses_ssb_tx_pulse_number = static_cast<int>(((0x1F) & ((p[62]) >> (0))));
  m_ses_ssb_signal_type = static_cast<int>(((0xF) & ((p[63]) >> (4))));
  m_ses_ssb_ignore_1 = static_cast<int>(((0x7) & ((p[63]) >> (1))));
  m_ses_ssb_swap = static_cast<int>(((0x1) & ((p[63]) >> (0))));
  m_ses_ssb_swath_number = static_cast<int>(((0xFF) & ((p[64]) >> (0))));
  m_number_of_quads = static_cast<int>(
      ((((0x1) * (p[66]))) + (((0x100) * (((0xFF) & (p[65])))))));
  m_ignore_4 = static_cast<int>(((0xFF) & ((p[67]) >> (0))));
}
CellItem_SpacePacketHeader0 &
CellItem_SpacePacketHeader0::operator=(const CellItem_SpacePacketHeader0 &src) {
  m_offset = src.m_offset;
  return *this;
}
ListStore_SpacePacketHeader0::ListStore_SpacePacketHeader0()
    : m_VBox(Gtk::ORIENTATION_VERTICAL, 8),
      m_Label("ListStore_SpacePacketHeader0") {
  set_title("Gtk::ListStore ListStore_SpacePacketHeader0");
  set_border_width(8);
  set_default_size(280, 250);
  add(m_VBox);
  m_VBox.pack_start(m_Label, Gtk::PACK_SHRINK);
  m_ScrolledWindow.set_shadow_type(Gtk::SHADOW_ETCHED_IN);
  m_ScrolledWindow.set_policy(Gtk::POLICY_NEVER, Gtk::POLICY_AUTOMATIC);
  m_VBox.pack_start(m_ScrolledWindow);
  create_model();
  m_TreeView.set_model(m_refListStore);
  m_TreeView.set_search_column(m_columns.offset.index());
  add_columns();
  m_ScrolledWindow.add(m_TreeView);
  show_all();
}
ListStore_SpacePacketHeader0::~ListStore_SpacePacketHeader0() {}
void ListStore_SpacePacketHeader0::create_model() {
  m_refListStore = Gtk::ListStore::create(m_columns);
  add_items();
  std::for_each(
      m_vecItems.begin(), m_vecItems.end(),
      sigc::mem_fun(*this, &ListStore_SpacePacketHeader0::liststore_add_item));
}
void ListStore_SpacePacketHeader0::add_columns() {
  m_TreeView.append_column("offset", m_columns.offset);
}
void ListStore_SpacePacketHeader0::add_items() {
  for (auto val : state._header_offset) {
    m_vecItems.push_back(CellItem_SpacePacketHeader0(val));
  }
}
void ListStore_SpacePacketHeader0::liststore_add_item(
    const CellItem_SpacePacketHeader0 &foo) {
  auto row = *(m_refListStore->append());
  row[m_columns.offset] = foo.m_offset;
}
int main(int argc, char **argv) {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  state._filename =
      "/media/sdb4/sar/sao_paulo/"
      "s1b-s6-raw-s-vv-20200824t214314-20200824t214345-023070-02bce0.dat";
  init_mmap(state._filename);
  init_collect_packet_headers();
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm.example");
  ListStore_SpacePacketHeader0 hw;
  app->run(hw);
}