
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

HelloWorld::HelloWorld() : m_button("_Hello World", true) {
  set_border_width(10);
  m_button.signal_clicked().connect(
      sigc::mem_fun(*this, &HelloWorld::on_button_clicked));
  add(m_button);
  m_button.show();
}
HelloWorld::~HelloWorld() {}
void HelloWorld::on_button_clicked() {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("button") << (" ")
      << (std::endl) << (std::flush);
}
int main(int argc, char **argv) {
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm.example");
  HelloWorld hw;
  app->run(hw);
}