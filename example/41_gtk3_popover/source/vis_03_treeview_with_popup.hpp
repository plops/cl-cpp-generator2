#ifndef VIS_03_TREEVIEW_WITH_POPUP_H
#define VIS_03_TREEVIEW_WITH_POPUP_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
;
#include <gtkmm.h>
;
// header
#include <gtkmm.h>
 ;
class TreeView_WithPopup : public Gtk::TreeView {
        public:
         TreeView_WithPopup ()  ;  
         ~TreeView_WithPopup ()  ;  
        protected:
        bool on_button_press_event (GdkEventButton* event)  ;  
        void on_menu_file_popup_generic ()  ;  
            struct ModelColumns : public Gtk::TreeModelColumnRecord {
                Gtk::TreeModelColumn<unsigned int> m_col_id;
                Gtk::TreeModelColumn<Glib::ustring> m_col_name;
                 ModelColumns ()    {
                        add(m_col_id);
                        add(m_col_name);
};
};
    const ModelColumns m_Columns;
        Glib::RefPtr<Gtk::ListStore> m_refTreeModel;
        Gtk::Menu m_Menu_Popup;
};
#endif