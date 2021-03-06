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
#include <gtkmm-plplot.h>
;
// header
#include <gtkmm.h>
#include <gtkmm-plplot.h>
 ;
class Window : public Gtk::Window {
        public:
         Window ()  ;  
         ~Window ()  ;  
        private:
        Gtk::PLplot::Canvas canvas;
        Gtk::Grid grid;
        Gtk::PLplot::Plot2D *plot=nullptr;
        Glib::ustring m_severity;
        Glib::ustring m_description;
        void add_plot_1 ()  ;  
};
int main (int argc, char** argv)  ;  
#endif