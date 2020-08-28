
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <gtkmm.h>
#include <iostream>

// implementation
#include "vis_00_base.hpp"
int main(int argc, char **argv) {
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm.examples.base");
  Gtk::Window win;
  win.set_default_size(200, 200);
  app->run(win);
}