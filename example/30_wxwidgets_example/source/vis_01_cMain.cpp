
#include "utils.h"

#include "globals.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <experimental/iterator>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <wx/wx.h>

// implementation
#include "vis_01_cMain.hpp"
// implementation
wxBEGIN_EVENT_TABLE(cMain, wxFrame) EVT_BUTTON(10001, cMain::OnButtonClicked)
    wxEND_EVENT_TABLE();
cMain::cMain()
    : wxFrame(nullptr, wxID_ANY, "title", wxPoint(30, 30), wxSize(800, 600)) {
  m_btn1 =
      new wxButton(this, 10001, "click me", wxPoint(10, 10), wxSize(150, 50));
  m_txt1 = new wxTextCtrl(this, wxID_ANY, "", wxPoint(10, 70), wxSize(300, 30));
  m_list1 = new wxListBox(this, wxID_ANY, wxPoint(10, 110), wxSize(300, 300));
}
cMain::~cMain() {}
void cMain::OnButtonClicked(wxCommandEvent &evt) {
  m_list1->AppendString(m_txt1->GetValue());
  evt.Skip();
}