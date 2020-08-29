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
class HelloWorld : public Gtk::Window {
        public:
         HelloWorld ()  ;  
         ~HelloWorld ()  ;  
        protected:
        Gtk::Button m_button;
};
int main (int argc, char** argv)  ;  
#endif