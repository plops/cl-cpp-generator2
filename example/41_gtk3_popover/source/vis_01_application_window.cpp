
#include "utils.h"

#include "globals.h"

extern State state;
#include "vis_03_treeview_with_popup.hpp"
#include <chrono>
#include <iostream>
#include <thread>
// implementation
#include "vis_01_application_window.hpp"

ExampleWindow::ExampleWindow() : m_VBox(Gtk::ORIENTATION_VERTICAL) {
  set_title("Gtk::TreeView example with popup");
  set_border_width(6);
  set_default_size(600, 400);
  add(m_VBox);
  m_ScrolledWindow.add(m_TreeView);
  m_ScrolledWindow.set_policy(Gtk::POLICY_AUTOMATIC, Gtk::POLICY_AUTOMATIC);
  m_VBox.pack_start(m_ScrolledWindow);
  show_all_children();
}
ExampleWindow::~ExampleWindow() {}