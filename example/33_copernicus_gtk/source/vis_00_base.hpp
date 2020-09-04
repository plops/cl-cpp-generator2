#ifndef VIS_00_BASE_H
#define VIS_00_BASE_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
#include <unordered_map>
;
#include <gtkmm-3.0/gtkmm/treeview.h>
#include <gtkmm-3.0/gtkmm/liststore.h>
#include <gtkmm-3.0/gtkmm/box.h>
#include <gtkmm-3.0/gtkmm/scrolledwindow.h>
#include <gtkmm-3.0/gtkmm/window.h>
;
// header
#include <gtkmm-3.0/gtkmm/treeview.h>
#include <gtkmm-3.0/gtkmm/liststore.h>
#include <gtkmm-3.0/gtkmm/box.h>
#include <gtkmm-3.0/gtkmm/scrolledwindow.h>
#include <gtkmm-3.0/gtkmm/window.h>
 ;
class CellItem_SpacePacketHeader0  {
        public:
         CellItem_SpacePacketHeader0 ()  ;  
         ~CellItem_SpacePacketHeader0 ()  ;  
         CellItem_SpacePacketHeader0 (const CellItem_SpacePacketHeader0& src)  ;  
         CellItem_SpacePacketHeader0 (gsize offset)  ;  
        CellItem_SpacePacketHeader0& operator= (const CellItem_SpacePacketHeader0& src)  ;  
        gsize m_offset;
};
class ListStore_SpacePacketHeader0 : public Gtk::Window {
        public:
         ListStore_SpacePacketHeader0 ()  ;  
         ~ListStore_SpacePacketHeader0 ()  ;  
        protected:
        void create_model ()  ;  
        void add_columns ()  ;  
        void add_items ()  ;  
        void liststore_add_item (const CellItem_SpacePacketHeader0& foo)  ;  
        Gtk::Box m_VBox;
        Gtk::ScrolledWindow m_ScrolledWindow;
        Gtk::Label m_Label;
        Gtk::TreeView m_TreeView;
        Glib::RefPtr<Gtk::ListStore> m_refListStore;
        typedef std::vector<CellItem_SpacePacketHeader0> type_vecItems;
        type_vecItems m_vecItems;
            struct ModelColumns : public Gtk::TreeModelColumnRecord {
                Gtk::TreeModelColumn<gsize> offset;
                 ModelColumns ()    {
                        add(offset);
};
};
    const ModelColumns m_columns;
};
int main (int argc, char** argv)  ;  
#endif