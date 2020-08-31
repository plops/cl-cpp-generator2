#ifndef VIS_00_BASE_H
#define VIS_00_BASE_H
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
class CellItem_Bug  {
        public:
         CellItem_Bug ()  ;  
         ~CellItem_Bug ()  ;  
         CellItem_Bug (const CellItem_Bug& src)  ;  
         CellItem_Bug (bool fixed, guint number, const Glib::ustring& severity, const Glib::ustring& description)  ;  
        CellItem_Bug& operator= (const CellItem_Bug& src)  ;  
        bool m_fixed;
        guint m_number;
        Glib::ustring m_severity;
        Glib::ustring m_description;
};
class Example_TreeView_ListStore : public Gtk::Window {
        public:
         Example_TreeView_ListStore ()  ;  
         ~Example_TreeView_ListStore ()  ;  
        protected:
        void create_model ()  ;  
        void add_columns ()  ;  
        void add_items ()  ;  
        void liststore_add_item (const CellItem_Bug& foo)  ;  
        Gtk::Box m_VBox;
        Gtk::ScrolledWindow m_ScrolledWindow;
        Gtk::Label m_Label;
        Gtk::TreeView m_TreeView;
        Glib::RefPtr<Gtk::ListStore> m_refListStore;
        typedef std::vector<CellItem_Bug> type_vecITems;
        type_vecItems m_vecItems;
        struct ModelColumns : public Gtk::TreeModelColumnRecord {
                Gtk::TreeModelColumn<bool> fixed;
                Gtk::TreeModelColumn<unsigned int> number;
                Gtk::TreeModelColumn<Glib::ustring> severity;
                Gtk::TreeModelColumn<Glib::usrting> description;
                 ModelColumns ()    {
                        add(fixed);
                        add(number);
                        add(severity);
                        add(description);
};
};
        const ModelColumns m_columns;
};
Gtk::Window* do_treeview_liststore ()  ;  
class HelloWorld : public Gtk::Window {
        public:
         HelloWorld ()  ;  
         ~HelloWorld ()  ;  
        protected:
        Gtk::Button m_button;
};
int main (int argc, char** argv)  ;  
#endif