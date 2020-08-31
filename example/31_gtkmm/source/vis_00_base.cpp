
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

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
  pColumn->set_fixed_width(50);
  pColumn->set_clickable();
  m_TreeView.append_column("Bug Number", m_columns.number);
  m_TreeView.append_column("Severity", m_columns.severity);
  m_TreeView.append_column("Description", m_columns.description);
}
void Example_TreeView_ListStore::add_items() {}
void Example_TreeView_ListStore::liststore_add_item(const CellItem_Bug &foo) {
  auto row = *(m_refListStore->append());
  row[m_columns.fixed] = foo.m_fixed;
  row[m_columns.number] = foo.m_number;
  row[m_columns.severity] = foo.m_severity;
  row[m_columns.description] = foo.m_description;
}
Gtk::Window *do_treeview_liststore() {
  return new Example_TreeView_ListStore();
}
HelloWorld::HelloWorld() : m_button("_Hello World", true) {
  set_border_width(10);
  m_button.signal_clicked().connect([]() {
    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("button") << (" ") << (std::endl) << (std::flush);
  });
  add(m_button);
  m_button.show();
}
HelloWorld::~HelloWorld() {}
int main(int argc, char **argv) {
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm.example");
  HelloWorld hw;
  app->run(hw);
}