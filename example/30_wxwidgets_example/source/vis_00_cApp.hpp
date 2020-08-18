#ifndef VIS_00_CAPP_H
#define VIS_00_CAPP_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
;
#include <wx/wx.h>
;
// header;
class cApp : public wxApp {
        public:
         cApp ()  ;  
         ~cApp ()  ;  
        public:
        virtual bool OnInit ()  ;  ;
};
#endif