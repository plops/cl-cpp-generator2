
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <map>
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
  m_packet_version_number = src.m_packet_version_number;
  m_packet_type = src.m_packet_type;
  m_secondary_header_flag = src.m_secondary_header_flag;
  m_application_process_id_process_id = src.m_application_process_id_process_id;
  m_application_process_id_packet_category =
      src.m_application_process_id_packet_category;
  m_sequence_flags = src.m_sequence_flags;
  m_sequence_count = src.m_sequence_count;
  m_data_length = src.m_data_length;
  m_coarse_time = src.m_coarse_time;
  m_fine_time = src.m_fine_time;
  m_sync_marker = src.m_sync_marker;
  m_data_take_id = src.m_data_take_id;
  m_ecc_number = src.m_ecc_number;
  m_ignore_0 = src.m_ignore_0;
  m_test_mode = src.m_test_mode;
  m_rx_channel_id = src.m_rx_channel_id;
  m_instrument_configuration_id = src.m_instrument_configuration_id;
  m_sub_commutated_index = src.m_sub_commutated_index;
  m_sub_commutated_data = src.m_sub_commutated_data;
  m_space_packet_count = src.m_space_packet_count;
  m_pri_count = src.m_pri_count;
  m_error_flag = src.m_error_flag;
  m_ignore_1 = src.m_ignore_1;
  m_baq_mode = src.m_baq_mode;
  m_baq_block_length = src.m_baq_block_length;
  m_ignore_2 = src.m_ignore_2;
  m_range_decimation = src.m_range_decimation;
  m_rx_gain = src.m_rx_gain;
  m_tx_ramp_rate_polarity = src.m_tx_ramp_rate_polarity;
  m_tx_ramp_rate_magnitude = src.m_tx_ramp_rate_magnitude;
  m_tx_pulse_start_frequency_polarity = src.m_tx_pulse_start_frequency_polarity;
  m_tx_pulse_start_frequency_magnitude =
      src.m_tx_pulse_start_frequency_magnitude;
  m_tx_pulse_length = src.m_tx_pulse_length;
  m_ignore_3 = src.m_ignore_3;
  m_rank = src.m_rank;
  m_pulse_repetition_interval = src.m_pulse_repetition_interval;
  m_sampling_window_start_time = src.m_sampling_window_start_time;
  m_sampling_window_length = src.m_sampling_window_length;
  m_sab_ssb_calibration_p = src.m_sab_ssb_calibration_p;
  m_sab_ssb_polarisation = src.m_sab_ssb_polarisation;
  m_sab_ssb_temp_comp = src.m_sab_ssb_temp_comp;
  m_sab_ssb_ignore_0 = src.m_sab_ssb_ignore_0;
  m_sab_ssb_elevation_beam_address = src.m_sab_ssb_elevation_beam_address;
  m_sab_ssb_ignore_1 = src.m_sab_ssb_ignore_1;
  m_sab_ssb_azimuth_beam_address = src.m_sab_ssb_azimuth_beam_address;
  m_ses_ssb_cal_mode = src.m_ses_ssb_cal_mode;
  m_ses_ssb_ignore_0 = src.m_ses_ssb_ignore_0;
  m_ses_ssb_tx_pulse_number = src.m_ses_ssb_tx_pulse_number;
  m_ses_ssb_signal_type = src.m_ses_ssb_signal_type;
  m_ses_ssb_ignore_1 = src.m_ses_ssb_ignore_1;
  m_ses_ssb_swap = src.m_ses_ssb_swap;
  m_ses_ssb_swath_number = src.m_ses_ssb_swath_number;
  m_number_of_quads = src.m_number_of_quads;
  m_ignore_4 = src.m_ignore_4;
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
  m_ScrolledWindow.set_policy(Gtk::POLICY_AUTOMATIC, Gtk::POLICY_AUTOMATIC);
  m_VBox.pack_start(m_ScrolledWindow);
  create_model();
  m_TreeView.set_model(m_refListStore);
  m_TreeView.set_search_column(m_columns.offset.index());
  m_TreeView.set_has_tooltip(true);
  m_TreeView.signal_query_tooltip().connect(
      [this](int x, int y, bool keyboard_tooltip,
             const Glib::RefPtr<Gtk::Tooltip> &tooltip) -> bool {
        Glib::ustring column_title;
        if (keyboard_tooltip) {
          Gtk::TreeModel::Path path;
          Gtk::TreeViewColumn *focus_column;
          m_TreeView.get_cursor(path, focus_column);
          if (!((nullptr) == (focus_column))) {
            column_title = focus_column->get_title();
          }
        } else {
          int bx = 0;
          int by = 0;
          m_TreeView.convert_widget_to_bin_window_coords(x, y, bx, by);
          Gtk::TreeModel::Path path;
          Gtk::TreeViewColumn *column;
          int cx = 0;
          int cy = 0;
          m_TreeView.get_path_at_pos(bx, by, path, column, cx, cy);
          if (!((nullptr) == (column))) {
            column_title = column->get_title();
          }
        }
        std::map<std::string, std::string> short_to_long_column_name = {
            {"pvn", "packet_version_number"},
            {"pt", "packet_type"},
            {"shf", "secondary_header_flag"},
            {"apipi", "application_process_id_process_id"},
            {"apipc", "application_process_id_packet_category"},
            {"sf", "sequence_flags"},
            {"sc", "sequence_count"},
            {"dl", "data_length"},
            {"ct", "coarse_time"},
            {"ft", "fine_time"},
            {"sm", "sync_marker"},
            {"dti", "data_take_id"},
            {"en", "ecc_number"},
            {"i0", "ignore_0"},
            {"tm", "test_mode"},
            {"rci", "rx_channel_id"},
            {"ici", "instrument_configuration_id"},
            {"sci", "sub_commutated_index"},
            {"scd", "sub_commutated_data"},
            {"spc", "space_packet_count"},
            {"pc", "pri_count"},
            {"ef", "error_flag"},
            {"i1", "ignore_1"},
            {"bm", "baq_mode"},
            {"bbl", "baq_block_length"},
            {"i2", "ignore_2"},
            {"rd", "range_decimation"},
            {"rg", "rx_gain"},
            {"trrp", "tx_ramp_rate_polarity"},
            {"trrm", "tx_ramp_rate_magnitude"},
            {"tpsfp", "tx_pulse_start_frequency_polarity"},
            {"tpsfm", "tx_pulse_start_frequency_magnitude"},
            {"tpl", "tx_pulse_length"},
            {"i3", "ignore_3"},
            {"r", "rank"},
            {"pri", "pulse_repetition_interval"},
            {"swst", "sampling_window_start_time"},
            {"swl", "sampling_window_length"},
            {"sscp", "sab_ssb_calibration_p"},
            {"ssp", "sab_ssb_polarisation"},
            {"sstc", "sab_ssb_temp_comp"},
            {"ssi0", "sab_ssb_ignore_0"},
            {"sseba", "sab_ssb_elevation_beam_address"},
            {"ssi1", "sab_ssb_ignore_1"},
            {"ssaba", "sab_ssb_azimuth_beam_address"},
            {"sscm", "ses_ssb_cal_mode"},
            {"ssi0", "ses_ssb_ignore_0"},
            {"sstpn", "ses_ssb_tx_pulse_number"},
            {"ssst", "ses_ssb_signal_type"},
            {"ssi1", "ses_ssb_ignore_1"},
            {"sss", "ses_ssb_swap"},
            {"sssn", "ses_ssb_swath_number"},
            {"noq", "number_of_quads"},
            {"i4", "ignore_4"}};
        auto long_column_name = short_to_long_column_name[column_title];
        tooltip->set_text(long_column_name);
        return true;
      });
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
  m_TreeView.append_column("pvn", m_columns.packet_version_number);
  m_TreeView.append_column("pt", m_columns.packet_type);
  m_TreeView.append_column("shf", m_columns.secondary_header_flag);
  m_TreeView.append_column("apipi",
                           m_columns.application_process_id_process_id);
  m_TreeView.append_column("apipc",
                           m_columns.application_process_id_packet_category);
  m_TreeView.append_column("sf", m_columns.sequence_flags);
  m_TreeView.append_column("sc", m_columns.sequence_count);
  m_TreeView.append_column("dl", m_columns.data_length);
  m_TreeView.append_column("ct", m_columns.coarse_time);
  m_TreeView.append_column("ft", m_columns.fine_time);
  m_TreeView.append_column("sm", m_columns.sync_marker);
  m_TreeView.append_column("dti", m_columns.data_take_id);
  m_TreeView.append_column("en", m_columns.ecc_number);
  m_TreeView.append_column("i0", m_columns.ignore_0);
  m_TreeView.append_column("tm", m_columns.test_mode);
  m_TreeView.append_column("rci", m_columns.rx_channel_id);
  m_TreeView.append_column("ici", m_columns.instrument_configuration_id);
  m_TreeView.append_column("sci", m_columns.sub_commutated_index);
  m_TreeView.append_column("scd", m_columns.sub_commutated_data);
  m_TreeView.append_column("spc", m_columns.space_packet_count);
  m_TreeView.append_column("pc", m_columns.pri_count);
  m_TreeView.append_column("ef", m_columns.error_flag);
  m_TreeView.append_column("i1", m_columns.ignore_1);
  m_TreeView.append_column("bm", m_columns.baq_mode);
  m_TreeView.append_column("bbl", m_columns.baq_block_length);
  m_TreeView.append_column("i2", m_columns.ignore_2);
  m_TreeView.append_column("rd", m_columns.range_decimation);
  m_TreeView.append_column("rg", m_columns.rx_gain);
  m_TreeView.append_column("trrp", m_columns.tx_ramp_rate_polarity);
  m_TreeView.append_column("trrm", m_columns.tx_ramp_rate_magnitude);
  m_TreeView.append_column("tpsfp",
                           m_columns.tx_pulse_start_frequency_polarity);
  m_TreeView.append_column("tpsfm",
                           m_columns.tx_pulse_start_frequency_magnitude);
  m_TreeView.append_column("tpl", m_columns.tx_pulse_length);
  m_TreeView.append_column("i3", m_columns.ignore_3);
  m_TreeView.append_column("r", m_columns.rank);
  m_TreeView.append_column("pri", m_columns.pulse_repetition_interval);
  m_TreeView.append_column("swst", m_columns.sampling_window_start_time);
  m_TreeView.append_column("swl", m_columns.sampling_window_length);
  m_TreeView.append_column("sscp", m_columns.sab_ssb_calibration_p);
  m_TreeView.append_column("ssp", m_columns.sab_ssb_polarisation);
  m_TreeView.append_column("sstc", m_columns.sab_ssb_temp_comp);
  m_TreeView.append_column("ssi0", m_columns.sab_ssb_ignore_0);
  m_TreeView.append_column("sseba", m_columns.sab_ssb_elevation_beam_address);
  m_TreeView.append_column("ssi1", m_columns.sab_ssb_ignore_1);
  m_TreeView.append_column("ssaba", m_columns.sab_ssb_azimuth_beam_address);
  m_TreeView.append_column("sscm", m_columns.ses_ssb_cal_mode);
  m_TreeView.append_column("ssi0", m_columns.ses_ssb_ignore_0);
  m_TreeView.append_column("sstpn", m_columns.ses_ssb_tx_pulse_number);
  m_TreeView.append_column("ssst", m_columns.ses_ssb_signal_type);
  m_TreeView.append_column("ssi1", m_columns.ses_ssb_ignore_1);
  m_TreeView.append_column("sss", m_columns.ses_ssb_swap);
  m_TreeView.append_column("sssn", m_columns.ses_ssb_swath_number);
  m_TreeView.append_column("noq", m_columns.number_of_quads);
  m_TreeView.append_column("i4", m_columns.ignore_4);
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
  row[m_columns.packet_version_number] = foo.m_packet_version_number;
  row[m_columns.packet_type] = foo.m_packet_type;
  row[m_columns.secondary_header_flag] = foo.m_secondary_header_flag;
  row[m_columns.application_process_id_process_id] =
      foo.m_application_process_id_process_id;
  row[m_columns.application_process_id_packet_category] =
      foo.m_application_process_id_packet_category;
  row[m_columns.sequence_flags] = foo.m_sequence_flags;
  row[m_columns.sequence_count] = foo.m_sequence_count;
  row[m_columns.data_length] = foo.m_data_length;
  row[m_columns.coarse_time] = foo.m_coarse_time;
  row[m_columns.fine_time] = foo.m_fine_time;
  row[m_columns.sync_marker] = foo.m_sync_marker;
  row[m_columns.data_take_id] = foo.m_data_take_id;
  row[m_columns.ecc_number] = foo.m_ecc_number;
  row[m_columns.ignore_0] = foo.m_ignore_0;
  row[m_columns.test_mode] = foo.m_test_mode;
  row[m_columns.rx_channel_id] = foo.m_rx_channel_id;
  row[m_columns.instrument_configuration_id] =
      foo.m_instrument_configuration_id;
  row[m_columns.sub_commutated_index] = foo.m_sub_commutated_index;
  row[m_columns.sub_commutated_data] = foo.m_sub_commutated_data;
  row[m_columns.space_packet_count] = foo.m_space_packet_count;
  row[m_columns.pri_count] = foo.m_pri_count;
  row[m_columns.error_flag] = foo.m_error_flag;
  row[m_columns.ignore_1] = foo.m_ignore_1;
  row[m_columns.baq_mode] = foo.m_baq_mode;
  row[m_columns.baq_block_length] = foo.m_baq_block_length;
  row[m_columns.ignore_2] = foo.m_ignore_2;
  row[m_columns.range_decimation] = foo.m_range_decimation;
  row[m_columns.rx_gain] = foo.m_rx_gain;
  row[m_columns.tx_ramp_rate_polarity] = foo.m_tx_ramp_rate_polarity;
  row[m_columns.tx_ramp_rate_magnitude] = foo.m_tx_ramp_rate_magnitude;
  row[m_columns.tx_pulse_start_frequency_polarity] =
      foo.m_tx_pulse_start_frequency_polarity;
  row[m_columns.tx_pulse_start_frequency_magnitude] =
      foo.m_tx_pulse_start_frequency_magnitude;
  row[m_columns.tx_pulse_length] = foo.m_tx_pulse_length;
  row[m_columns.ignore_3] = foo.m_ignore_3;
  row[m_columns.rank] = foo.m_rank;
  row[m_columns.pulse_repetition_interval] = foo.m_pulse_repetition_interval;
  row[m_columns.sampling_window_start_time] = foo.m_sampling_window_start_time;
  row[m_columns.sampling_window_length] = foo.m_sampling_window_length;
  row[m_columns.sab_ssb_calibration_p] = foo.m_sab_ssb_calibration_p;
  row[m_columns.sab_ssb_polarisation] = foo.m_sab_ssb_polarisation;
  row[m_columns.sab_ssb_temp_comp] = foo.m_sab_ssb_temp_comp;
  row[m_columns.sab_ssb_ignore_0] = foo.m_sab_ssb_ignore_0;
  row[m_columns.sab_ssb_elevation_beam_address] =
      foo.m_sab_ssb_elevation_beam_address;
  row[m_columns.sab_ssb_ignore_1] = foo.m_sab_ssb_ignore_1;
  row[m_columns.sab_ssb_azimuth_beam_address] =
      foo.m_sab_ssb_azimuth_beam_address;
  row[m_columns.ses_ssb_cal_mode] = foo.m_ses_ssb_cal_mode;
  row[m_columns.ses_ssb_ignore_0] = foo.m_ses_ssb_ignore_0;
  row[m_columns.ses_ssb_tx_pulse_number] = foo.m_ses_ssb_tx_pulse_number;
  row[m_columns.ses_ssb_signal_type] = foo.m_ses_ssb_signal_type;
  row[m_columns.ses_ssb_ignore_1] = foo.m_ses_ssb_ignore_1;
  row[m_columns.ses_ssb_swap] = foo.m_ses_ssb_swap;
  row[m_columns.ses_ssb_swath_number] = foo.m_ses_ssb_swath_number;
  row[m_columns.number_of_quads] = foo.m_number_of_quads;
  row[m_columns.ignore_4] = foo.m_ignore_4;
}
void create_draw_area() {
  Gtk::DrawingArea area;
  auto ctx = area.get_window()->create_cairo_context();
  ctx->set_source_rgb((1.0f), (0.f), (0.f));
  ctx->set_line_width((2.0f));
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