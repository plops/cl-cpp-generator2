
#include "utils.h"

#include "globals.h"

extern State state;
#include <FL/Fl.h>
#include <FL/Fl_Box.h>
#include <FL/Fl_Window.h>
#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv='") << (argv) << ("'") << (std::endl) << (std::flush);
  auto win = new Fl_Window(200, 200, "window title");
  auto box = new Fl_Box(0, 0, 200, 200, "hello world");
  win->show();
  return Fl::run();
}