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
#include <FL/Fl.h>
#include <FL/Fl_Box.h>
#include <FL/Fl_Window.h>
;
class Window : public Fl_Window {
        public:
         Window (int w, int h, const string& title)  ;  
         Window (Point xy, int w, int h, const string& title)  ;  
         ~Window ()  ;  
};
class Widget  {
        public:
         Widget (Point xy, int w, int h, const string& s)  ;  
         ~Widget ()  ;  
        void move (int dx, int dy)  ;  
        void hide ()  ;  
        void show ()  ;  
        virtual void attach(Window&)=0;
        protected:
        Window* own;
        Fl_Widget* pw;
};
int main (int argc, char** argv)  ;  
#endif