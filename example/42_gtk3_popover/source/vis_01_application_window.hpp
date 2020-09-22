#ifndef VIS_01_APPLICATION_WINDOW_H
#define VIS_01_APPLICATION_WINDOW_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
;
#include "vis_03_treeview_with_popup.hpp"
;
#include <gtkmm/window.h>
#include <gtkmm/grid.h>
;
#include <gtkmm/box.h>
#include <gtkmm/scrolledwindow.h>
#include <gtkmm/widget.h>
#include <gtkmm/cssprovider.h>
#include <gtkmm/styleproperty.h>
;
#include <glibmm/extraclassinit.h>
#include <glibmm/ustring.h>
;
#include <gdkmm/general.h>
;
// header
#include <gtkmm/window.h>
#include <gtkmm/grid.h>
#include <gtkmm/box.h>
#include <gtkmm/scrolledwindow.h>
#include <gtkmm/widget.h>
#include <gtkmm/cssprovider.h>
#include <gtkmm/styleproperty.h>
#include <glibmm/extraclassinit.h>
#include <glibmm/ustring.h>
#include <gdkmm/general.h>
 ;
class ExampleWindow : public Gtk::Window {
        public:
         ExampleWindow ()  ;  
         ~ExampleWindow ()  ;  
        Gtk::Box m_VBox;
        Gtk::ScrolledWindow m_ScrolledWindow;
        TreeView_WithPopup m_TreeView;
};
#endif