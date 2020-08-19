#ifndef VIS_01_CMAIN_H
#define VIS_01_CMAIN_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <cstdio>
#include <cassert>
#include <string>
#include <fstream>
#include <thread>
#include <vector>
#include <experimental/iterator>
#include <algorithm>
;
#include <wx/wx.h>
;
// header;
// header;
class cMain : public wxFrame {
        public:
         cMain ()  ;  
         ~cMain ()  ;  
        public:
                int button_field_n=10;
    int button_field_m=10;
    wxButton** btn=nullptr;
        void OnButtonClicked (wxCommandEvent& evt)  ;  
        wxDECLARE_EVENT_TABLE();
};
#endif