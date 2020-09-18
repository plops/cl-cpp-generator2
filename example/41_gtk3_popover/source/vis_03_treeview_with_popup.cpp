
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>
// implementation
#include "vis_03_treeview_with_popup.hpp"

TreeView_WithPopup::TreeView_WithPopup() {
  m_refTreeModel = Gtk::ListStore::create(m_Columns);
  set_model(m_refTreeModel);
}
TreeView_WithPopup::~TreeView_WithPopup() {}
bool TreeView_WithPopup::on_button_press_event(GdkEventButton *event) {
  auto return_value = false;
  return return_value;
}
void TreeView_WithPopup::on_menu_file_popup_generic() {}