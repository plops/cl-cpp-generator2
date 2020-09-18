#ifndef VIS_02_DRAWING_WIDGET_H
#define VIS_02_DRAWING_WIDGET_H
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
#include <gtkmm/widget.h>
#include <gtkmm/cssprovider.h>
#include <gtkmm/styleproperty.h>
#include <glibmm/extraclassinit.h>
#include <glibmm/ustring.h>
#include <gdkmm/general.h>
 ;
class PenroseWidget : public Gtk::Widget {
        public:
         PenroseWidget ()  ;  
         ~PenroseWidget ()  ;  
        protected:
        Gtk::SizeRequestMode get_request_mode_vfunc ()  ;  
        void get_preferred_width_vfunc (int& minimum_width, int& natural_width)  ;  
        void get_preferred_height_for_width_vfunc (int width, int& minimum_height, int& natural_height)  ;  
        void get_preferred_height_vfunc (int& minimum_height, int& natural_height)  ;  
        void get_preferred_width_for_height_vfunc (int height, int& minimum_height, int& natural_height)  ;  
        void on_size_allocate (Gtk::Allocation& allocation)  ;  
        void on_map ()  ;  
        void on_unmap ()  ;  
        void on_realize ()  ;  
        void on_unrealize ()  ;  
        bool on_draw (const Cairo::RefPtr<Cairo::Context>& cr)  ;  
        void on_parsing_error (const Glib::RefPtr<const Gtk::CssSection>& section, const Glib::Error& error)  ;  
        Gtk::StyleProperty<int> m_scale_prop;
        Glib::RefPtr<Gdk::Window> m_refGdkWindow;
        Glib::RefPtr<Gtk::CssProvider> m_refCssProvider;
        int m_scale;
};
#endif