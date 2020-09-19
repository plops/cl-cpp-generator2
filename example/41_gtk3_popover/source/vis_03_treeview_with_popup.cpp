
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
  {
    auto item = Gtk::make_managed<Gtk::MenuItem>("_Edit", true);
    item->signal_activate().connect(
        sigc::mem_fun(*this, &TreeView_WithPopup::on_menu_file_popup_generic));
    m_Menu_Popup.append(*item);
  }
  {
    auto item = Gtk::make_managed<Gtk::MenuItem>("_Process", true);
    item->signal_activate().connect(
        sigc::mem_fun(*this, &TreeView_WithPopup::on_menu_file_popup_generic));
    m_Menu_Popup.append(*item);
  }
  {
    auto item = Gtk::make_managed<Gtk::MenuItem>("_Remove", true);
    item->signal_activate().connect(
        sigc::mem_fun(*this, &TreeView_WithPopup::on_menu_file_popup_generic));
    m_Menu_Popup.append(*item);
  }
  m_Menu_Popup.accelerate(*this);
  m_Menu_Popup.show_all();
}
TreeView_WithPopup::~TreeView_WithPopup() {}
bool TreeView_WithPopup::on_button_press_event(GdkEventButton *event) {
  auto return_value = false;
  return_value = TreeView::on_button_press_event(event);
  if ((((GDK_BUTTON_PRESS) == (event->type)) && ((3) == (event->button)))) {
    m_Menu_Popup.popup_at_pointer(reinterpret_cast<GdkEvent *>(event));
  }
  return return_value;
}
void TreeView_WithPopup::on_menu_file_popup_generic() {
  auto sel = get_selection();
  if (sel) {
    auto iter = sel->get_selected();
    if (iter) {
      auto id = (*iter)[m_Columns.m_col_id];

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("popup was selected") << (" ") << (std::setw(8))
                  << (" id='") << (id) << ("'") << (std::endl) << (std::flush);
    }
  }
}