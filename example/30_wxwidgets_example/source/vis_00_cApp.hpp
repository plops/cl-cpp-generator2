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
#include "vis_01_cMain.hpp"
;
// header;
class cApp : public wxApp {
        public:
         cApp ()  ;  
         ~cApp ()  ;  
        private:
            cMain* m_frame1=nullptr;
        public:
        bool OnInit ()  ;  
};
#endif