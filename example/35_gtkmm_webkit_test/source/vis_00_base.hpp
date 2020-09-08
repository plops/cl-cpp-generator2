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
#include <webkit2/webkit2.h>
;
// header
#include <gtkmm.h>
#include <webkit2/webkit2.h>
 ;
class Window : public Gtk::Widget {
        public:
         Window ()  ;  
         ~Window ()  ;  
         operator WebKitWebView* ()  ;  
        void load_uri (const gchar* uri)  ;  
};
int main (int argc, char** argv)  ;  
#endif