
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
  btn = new wxButton *[((button_field_n) * (button_field_m))];
  auto grid = new wxGridSizer(button_field_n, button_field_m, 0, 0);
  for (auto i = 0; (i) < (button_field_n); (i) += (1)) {
    for (auto j = 0; (j) < (button_field_m); (j) += (1)) {
      auto pos = ((i) + (((j) * (button_field_n))));
      btn[pos] = new wxButton(this, ((20000) + (pos)));
      grid->Add(btn[pos], ((wxEXPAND) | (wxALL)));
    }
  }
  this->SetSizer(grid);
}
cMain::~cMain() {}
void cMain::OnButtonClicked(wxCommandEvent &evt) { evt.Skip(); }