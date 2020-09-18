
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>
// implementation
#include "vis_01_application_window.hpp"

ExampleWindow::ExampleWindow() {
  set_title("custom widget example");
  set_border_width(6);
  set_default_size(600, 400);
  show_all_children();
}
ExampleWindow::~ExampleWindow() {}