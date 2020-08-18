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
class cMain : public wxFrame {
        public:
         cMain ()  ;  
         ~cMain ()  ;  
        public:
            wxButton *m_btn1=nullptr;
            wxTextCtrl *m_txt1=nullptr;
            wxListBox *m_list1=nullptr;
};
#endif