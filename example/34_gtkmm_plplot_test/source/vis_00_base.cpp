
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

int main(int argc, char **argv) {
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm.examples.base");
  Gtk::Window win;
  win.set_default_size(200, 200);
  app->run(win);
}
Window::Window() : canvas() {}
Window::~Window() {}
int main(int argc, char **argv) {
  Glib::set_application_name("gtkmm-plplot-test13");
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm-plplot.example");
  Window win;
  app->run(win);
}