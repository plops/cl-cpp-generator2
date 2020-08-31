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
#include <gtkmm/button.h>
#include <gtkmm/window.h>
;
// header
#include <gtkmm/button.h>
#include <gtkmm/window.h>
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
class HelloWorld : public Gtk::Window {
        public:
         HelloWorld ()  ;  
         ~HelloWorld ()  ;  
        protected:
        Gtk::Button m_button;
};
int main (int argc, char** argv)  ;  
#endif