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
#include <epoxy/gl.h>
#include <gtk/gtk.h>
#include <gdk/gdkx.h>
#include <GL/gl.h>
;
// header
#include <gtkmm.h>
#include <epoxy/gl.h>
#include <gtk/gtk.h>
#include <gdk/gdkx.h>
#include <GL/gl.h>
 ;
class GraphicsArea : public Gtk::Window {
        public:
         GraphicsArea ()  ;  
         ~GraphicsArea ()  ;  
        void run ()  ;  
        void onNotifcationFromThread ()  ;  
        bool render (const Glib::RefPtr<Gdk::GLContext>& ctx)  ;  
        public:
        Gtk::GLArea area;
        Gtk::Box vbox{Gtk::ORIENTATION_VERTICAL,false};
        private:
        Glib::Dispatcher dispatcher;
};
int main (int argc, char** argv)  ;  
#endif