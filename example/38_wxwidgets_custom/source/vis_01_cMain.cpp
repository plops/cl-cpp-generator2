
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
Widget::Widget(wxPanel *parent, int id)
    : wxPanel(parent, id, wxDefaultPosition, wxSize(-1, 30), wxSUNKEN_BORDER),
      m_parent(parent) {
  Connect(wxEVT_PAINT, wxPaintEventHandler(Widget::OnPaint));
  Connect(wxEVT_SIZE, wxSizeEventHandler(Widget::OnSize));
}
void Widget::OnSize(wxSizeEvent &event) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("size") << (" ")
      << (std::setw(8)) << (" event.GetId()='") << (event.GetId()) << ("'")
      << (std::endl) << (std::flush);
  Refresh();
}
void Widget::OnPaint(wxPaintEvent &event) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("paint") << (" ")
      << (std::setw(8)) << (" event.GetId()='") << (event.GetId()) << ("'")
      << (std::endl) << (std::flush);
}
// implementation
wxBEGIN_EVENT_TABLE(cMain, wxFrame) EVT_BUTTON(10001, cMain::OnButtonClicked)
    wxEND_EVENT_TABLE();
cMain::cMain()
    : wxFrame(nullptr, wxID_ANY, "title", wxPoint(30, 30), wxSize(800, 600)) {
  m_btn1 =
      new wxButton(this, 10001, "click me", wxPoint(10, 10), wxSize(150, 50));
  m_txt1 = new wxTextCtrl(this, wxID_ANY, "", wxPoint(10, 70), wxSize(300, 30));
  m_list1 = new wxListBox(this, wxID_ANY, wxPoint(10, 110), wxSize(300, 300));
  btn = new wxButton *[((button_field_n) * (button_field_m))];
  auto grid = new wxGridSizer(button_field_n, button_field_m, 0, 0);
  for (auto i = 0; (i) < (button_field_n); (i) += (1)) {
    for (auto j = 0; (j) < (button_field_m); (j) += (1)) {
      auto pos = ((i) + (((j) * (button_field_n))));
      btn[pos] = new wxButton(this, ((20000) + (pos)));
      grid->Add(btn[pos], ((wxEXPAND) | (wxALL)));
      btn[pos]->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &cMain::OnButtonClicked,
                     this);
    }
  }
  this->SetSizer(grid);
  grid->Layout();
}
cMain::~cMain() { delete[] btn; }
void cMain::OnButtonClicked(wxCommandEvent &evt) {
  auto id = evt.GetId();
  if ((id) < (19000)) {
    m_list1->AppendString(m_txt1->GetValue());
  } else {
    auto x = ((id) - (20000)) % button_field_n;
    auto y = ((((id) - (20000))) / (button_field_m));
    m_list1->AppendString("button");
  }
  evt.Skip();
}