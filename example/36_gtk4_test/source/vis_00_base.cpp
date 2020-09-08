
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

int main(int argc, char **argv) {
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm.example");
  Example_TreeView_ListStore hw;
  app->run(hw);
}