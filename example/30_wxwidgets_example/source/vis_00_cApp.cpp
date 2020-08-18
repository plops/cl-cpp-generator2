
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <wx/wx.h>

#include "vis_01_cMain.hpp"
// implementation
#include "vis_00_cApp.hpp"
wxIMPLEMENT_APP(cApp);
cApp::cApp() {}
cApp::~cApp() {}
bool cApp::OnInit() {
  m_frame1 = new cMain;
  m_frame1->Show();
  return true;
}