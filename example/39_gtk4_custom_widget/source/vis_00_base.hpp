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
#include <gtkmm/window.h>
#include <gtkmm/grid.h>
;
#include <gtkmm/widget.h>
#include <gtkmm/cssprovider.h>
;
#include <glibmm/extraclassinit.h>
#include <glibmm/ustring.h>
;
// header
#include <gtkmm/window.h>
#include <gtkmm/grid.h>
#include <gtkmm/widget.h>
#include <gtkmm/cssprovider.h>
#include <glibmm/extraclassinit.h>
#include <glibmm/ustring.h>
 ;
class PenroseExtraInit : public Glib::ExtraClassInit {
        public:
         PenroseExtraInit (const Glib::ustring& css_name)  ;  
        private:
        Glib::ustring m_css_name;
};
class PenroseWidget : public PenroseExtraInit, public Gtk::Widget {
        public:
         PenroseWidget ()  ;  
         ~PenroseWidget ()  ;  
        protected:
        Gtk::SizeRequestMode get_request_mode_vfunc ()  ;  
        void measure_vfunc (Gtk::Orientation orientation, int for_size, int& minimum, int& natural, int& minimum_baseline, int& natural_baseline)  ;  
        void on_map ()  ;  
        void on_unmap ()  ;  
        void on_realize ()  ;  
        void on_unrealize ()  ;  
        void snapshot_vfunc (const Glib::RefPtr<Gtk::Snapshot>& snapshot)  ;  
        void on_parsing_error (const Glib::RefPtr<Gtk::CssSection>& section, const Glib::Error& error)  ;  
        Gtk::Border m_padding;
        Glib::RefPtr<Gtk::CssProvider> m_refCssProvider;
};
class ExampleWindow : public Gtk::Window {
        public:
         ExampleWindow ()  ;  
         ~ExampleWindow ()  ;  
        Gtk::Grid m_grid;
        PenroseWidget m_penrose;
};
int main (int argc, char** argv)  ;  
#endif