
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
    : m_offset(offset) {}
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