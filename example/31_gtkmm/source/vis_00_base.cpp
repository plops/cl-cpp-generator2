
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

CellItem_Bug::CellItem_Bug() : m_fixed(false), m_number(0) {}
CellItem_Bug::~CellItem_Bug() {}
CellItem_Bug::CellItem_Bug(const CellItem_Bug &src) { operator=(src); }
CellItem_Bug::CellItem_Bug(bool fixed, guint number,
                           const Glib::ustring &severity,
                           const Glib::ustring &description)
    : m_fixed(fixed), m_number(number), m_severity(severity),
      m_description(description) {}
CellItem_Bug &CellItem_Bug::operator=(const CellItem_Bug &src) {
  m_fixed = src.m_fixed;
  m_number = src.m_number;
  m_severity = src.m_severity;
  m_description = src.m_description;
  return *this;
}
HelloWorld::HelloWorld() : m_button("_Hello World", true) {
  set_border_width(10);
  m_button.signal_clicked().connect([]() {
    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("button") << (" ") << (std::endl) << (std::flush);
  });
  add(m_button);
  m_button.show();
}
HelloWorld::~HelloWorld() {}
int main(int argc, char **argv) {
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm.example");
  HelloWorld hw;
  app->run(hw);
}