
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
CellItem_SpacePacket::CellItem_SpacePacket() : m_swst(0) {}
CellItem_SpacePacket::~CellItem_SpacePacket() {}
CellItem_SpacePacket::CellItem_SpacePacket(const CellItem_SpacePacket &src) {
  operator=(src);
}
CellItem_SpacePacket::CellItem_SpacePacket(guint swst) : m_swst(swst) {}
CellItem_SpacePacket &
CellItem_SpacePacket::operator=(const CellItem_SpacePacket &src) {
  m_swst = src.m_swst;
  return *this;
}
CellItem_Bug::CellItem_Bug() : m_fixed(false), m_number(0) {}
CellItem_Bug::~CellItem_Bug() {}
CellItem_Bug::CellItem_Bug(const CellItem_Bug &src) { operator=(src); }
CellItem_Bug::CellItem_Bug(bool fixed, guint number,
                           const Glib::ustring &severity,
                           const Glib::ustring &description)
    : m_fixed(fixed), m_number(number), m_severity(severity),
      m_description(description) {}
CellItem_Bug &CellItem_Bug::operator=(const CellItem_Bug &src) {
  m_fixed = src.m_fixed;
  m_number = src.m_number;
  m_severity = src.m_severity;
  m_description = src.m_description;
  return *this;
}
Example_TreeView_ListStore::Example_TreeView_ListStore()
    : m_VBox(Gtk::ORIENTATION_VERTICAL, 8), m_Label("This is the bug list.") {
  set_title("Gtk::ListStore demo");
  set_border_width(8);
  set_default_size(280, 250);
  add(m_VBox);
  m_VBox.pack_start(m_Label, Gtk::PACK_SHRINK);
  m_ScrolledWindow.set_shadow_type(Gtk::SHADOW_ETCHED_IN);
  m_ScrolledWindow.set_policy(Gtk::POLICY_NEVER, Gtk::POLICY_AUTOMATIC);
  m_VBox.pack_start(m_ScrolledWindow);
  create_model();
  m_TreeView.set_model(m_refListStore);
  m_TreeView.set_search_column(m_columns.description.index());
  add_columns();
  m_ScrolledWindow.add(m_TreeView);
  show_all();
}
Example_TreeView_ListStore::~Example_TreeView_ListStore() {}
void Example_TreeView_ListStore::create_model() {
  m_refListStore = Gtk::ListStore::create(m_columns);
  add_items();
  std::for_each(
      m_vecItems.begin(), m_vecItems.end(),
      sigc::mem_fun(*this, &Example_TreeView_ListStore::liststore_add_item));
}
void Example_TreeView_ListStore::add_columns() {
  auto cols_count =
      m_TreeView.append_column_editable("Fixed?", m_columns.fixed);
  auto pColumn = m_TreeView.get_column(((cols_count) - (1)));
  pColumn->set_sizing(Gtk::TREE_VIEW_COLUMN_FIXED);
  pColumn->set_fixed_width(60);
  pColumn->set_clickable();
  m_TreeView.append_column("Bug Number", m_columns.number);
  m_TreeView.append_column("Severity", m_columns.severity);
  m_TreeView.append_column("Description", m_columns.description);
}
void Example_TreeView_ListStore::add_items() {
  m_vecItems.push_back(
      CellItem_Bug(false, 60482, "Normal", "scrollable notebuooks"));
  m_vecItems.push_back(CellItem_Bug(false, 60539, "Major", "trisatin"));
}
void Example_TreeView_ListStore::liststore_add_item(const CellItem_Bug &foo) {
  auto row = *(m_refListStore->append());
  row[m_columns.fixed] = foo.m_fixed;
  row[m_columns.number] = foo.m_number;
  row[m_columns.severity] = foo.m_severity;
  row[m_columns.description] = foo.m_description;
}
int main(int argc, char **argv) {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  state._filename =
      "/media/sdb4/sar/sao_paulo/"
      "s1b-s6-raw-s-vv-20200824t214314-20200824t214345-023070-02bce0.dat";
  init_mmap(state._filename);
  init_collect_packet_headers();
  auto packet_idx = 0;
  std::unordered_map<int, int> map_ele;
  std::unordered_map<int, int> map_cal;
  std::unordered_map<int, int> map_sig;
  auto cal_count = 0;
  init_sub_commutated_data_decoder();
  remove("./o_anxillary.csv");
  for (auto &e : state._header_data) {
    auto offset = state._header_offset[packet_idx];
    auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
    auto cal_p = ((0x1) & ((p[59]) >> (7)));
    auto ele = ((0xF) & ((p[60]) >> (4)));
    auto cal_type = ((ele) & (7));
    auto number_of_quads =
        ((((0x1) * (p[66]))) + (((0x100) * (((0xFF) & (p[65]))))));
    auto baq_mode = ((0x1F) & ((p[37]) >> (0)));
    auto test_mode = ((0x7) & ((p[21]) >> (4)));
    auto space_packet_count =
        ((((0x1) * (p[32]))) + (((0x100) * (p[31]))) + (((0x10000) * (p[30]))) +
         (((0x1000000) * (((0xFF) & (p[29]))))));
    auto sub_index = ((0xFF) & ((p[26]) >> (0)));
    auto sub_data = ((((0x1) * (p[28]))) + (((0x100) * (((0xFF) & (p[27]))))));
    auto signal_type = ((0xF) & ((p[63]) >> (4)));
    feed_sub_commutated_data_decoder(static_cast<uint16_t>(sub_data), sub_index,
                                     space_packet_count);
    (map_sig[signal_type])++;
    if (cal_p) {
      (cal_count)++;
      (map_cal[((ele) & (7))])++;

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("cal") << (" ") << (std::setw(8)) << (" cal_p='")
                  << (cal_p) << ("'") << (std::setw(8)) << (" cal_type='")
                  << (cal_type) << ("'") << (std::setw(8))
                  << (" number_of_quads='") << (number_of_quads) << ("'")
                  << (std::setw(8)) << (" baq_mode='") << (baq_mode) << ("'")
                  << (std::setw(8)) << (" test_mode='") << (test_mode) << ("'")
                  << (std::endl) << (std::flush);
    } else {
      (map_ele[ele]) += (number_of_quads);
    }
    (packet_idx)++;
  };
  for (auto &cal : map_cal) {
    auto number_of_cal = cal.second;
    auto cal_type = cal.first;

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("map_ele") << (" ") << (std::setw(8)) << (" cal_type='")
                << (cal_type) << ("'") << (std::setw(8)) << (" number_of_cal='")
                << (number_of_cal) << ("'") << (std::endl) << (std::flush);
  };
  for (auto &sig : map_sig) {
    auto number_of_sig = sig.second;
    auto sig_type = sig.first;

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("map_sig") << (" ") << (std::setw(8)) << (" sig_type='")
                << (sig_type) << ("'") << (std::setw(8)) << (" number_of_sig='")
                << (number_of_sig) << ("'") << (std::endl) << (std::flush);
  };
  auto ma = (-1.0f);
  auto ma_ele = -1;
  for (auto &elevation : map_ele) {
    auto number_of_Mquads =
        ((static_cast<float>(elevation.second)) / ((1.0e+6f)));
    auto elevation_beam_address = elevation.first;
    if ((ma) < (number_of_Mquads)) {
      ma = number_of_Mquads;
      ma_ele = elevation_beam_address;
    }

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("map_ele") << (" ") << (std::setw(8))
                << (" elevation_beam_address='") << (elevation_beam_address)
                << ("'") << (std::setw(8)) << (" number_of_Mquads='")
                << (number_of_Mquads) << ("'") << (std::endl) << (std::flush);
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("largest ele") << (" ")
      << (std::setw(8)) << (" ma_ele='") << (ma_ele) << ("'") << (std::setw(8))
      << (" ma='") << (ma) << ("'") << (std::setw(8)) << (" cal_count='")
      << (cal_count) << ("'") << (std::endl) << (std::flush);
  auto mi_data_delay = 10000000;
  auto ma_data_delay = -1;
  auto ma_data_end = -1;
  auto ele_number_echoes = 0;
  {
    std::unordered_map<int, int> map_azi;
    auto packet_idx2 = 0;
    for (auto &e : state._header_data) {
      auto offset = state._header_offset[packet_idx2];
      auto p = ((offset) + (static_cast<uint8_t *>(state._mmap_data)));
      auto ele = ((0xF) & ((p[60]) >> (4)));
      auto azi = ((((0x1) * (p[61]))) + (((0x100) * (((0x3) & (p[60]))))));
      auto number_of_quads =
          ((((0x1) * (p[66]))) + (((0x100) * (((0xFF) & (p[65]))))));
      auto cal_p = ((0x1) & ((p[59]) >> (7)));
      auto data_delay = ((40) + (((((0x1) * (p[55]))) + (((0x100) * (p[54]))) +
                                  (((0x10000) * (((0xFF) & (p[53]))))))));
      if (!(cal_p)) {
        if ((ele) == (ma_ele)) {
          (ele_number_echoes)++;
          if ((data_delay) < (mi_data_delay)) {
            mi_data_delay = data_delay;
          }
          if ((ma_data_delay) < (data_delay)) {
            ma_data_delay = data_delay;
          }
          auto v = ((data_delay) + (((2) * (number_of_quads))));
          if ((ma_data_end) < (v)) {
            ma_data_end = v;
          }
          (map_azi[azi]) += (number_of_quads);
        }
      }
      (packet_idx2)++;
    };

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("data_delay") << (" ") << (std::setw(8))
                << (" mi_data_delay='") << (mi_data_delay) << ("'")
                << (std::setw(8)) << (" ma_data_delay='") << (ma_data_delay)
                << ("'") << (std::setw(8)) << (" ma_data_end='")
                << (ma_data_end) << ("'") << (std::setw(8))
                << (" ele_number_echoes='") << (ele_number_echoes) << ("'")
                << (std::endl) << (std::flush);
    for (auto &azi : map_azi) {
      auto number_of_Mquads = ((static_cast<float>(azi.second)) / ((1.0e+6f)));
      auto azi_beam_address = azi.first;

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("map_azi") << (" ") << (std::setw(8))
                  << (" azi_beam_address='") << (azi_beam_address) << ("'")
                  << (std::setw(8)) << (" number_of_Mquads='")
                  << (number_of_Mquads) << ("'") << (std::endl) << (std::flush);
    };
  }
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm.example");
  Example_TreeView_ListStore hw;
  app->run(hw);
}