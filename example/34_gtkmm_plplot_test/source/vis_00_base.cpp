
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

Window::Window() : canvas() {}
Window::~Window() {}
int main(int argc, char **argv) {
  Glib::set_application_name("gtkmm-plplot-test13");
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm-plplot.example");
  Window win;
  app->run(win);
}