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
#include <giomm/appinfo.h>
#include <giomm/liststore.h>
;
// header
#include <gtkmm.h>
#include <giomm/appinfo.h>
#include <giomm/liststore.h>
 ;
class Example_ListView_AppLauncher : public Gtk::Window {
        public:
         Example_ListView_AppLauncher ()  ;  
         ~Example_ListView_AppLauncher ()  ;  
        protected:
        Glib::RefPtr<Gio::ListModel> create_application_list ()  ;  
        void setup_listitem (const Glib::RefPtr<Gtk::ListItem>& item)  ;  
        void bind_listitem (const Glib::RefPtr<Gtk::ListItem>& item)  ;  
        void activate (guint position)  ;  
        Gtk::ListView* m_list;
};
int main (int argc, char** argv)  ;  
#endif