
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
  {
    auto row = *(m_refTreeModel->append());
    row[m_Columns.m_col_id] = 1;
    row[m_Columns.m_col_name] = "right-click on this";
  }
  {
    auto row = *(m_refTreeModel->append());
    row[m_Columns.m_col_id] = 2;
    row[m_Columns.m_col_name] = "or this";
  }
  {
    auto row = *(m_refTreeModel->append());
    row[m_Columns.m_col_id] = 3;
    row[m_Columns.m_col_name] = "or this, for popup context menu";
  }
  append_column("id", m_Columns.m_col_id);
  append_column("name", m_Columns.m_col_name);
}
TreeView_WithPopup::~TreeView_WithPopup() {}
bool TreeView_WithPopup::on_button_press_event(GdkEventButton *event) {
  auto return_value = false;
  return return_value;
}
void TreeView_WithPopup::on_menu_file_popup_generic() {}