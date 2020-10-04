
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <nanogui/nanogui.h>
#include <thread>

// implementation
using namespace nanogui;
enum test_enum {
  Item1 = 0,
  Item2,
  Item3,
};
bool bvar = true;
int ivar = 1234;
double dvar = 21.34929e-3;
std::string strval = "a string";
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  nanogui::init();
  auto screen = new Screen(Vector2i(500, 700), "nanogui test gl4.1", true,
                           false, true, true, false, 4, 1);
  auto gui = new FormHelper(screen);
  auto window = gui->add_window(Vector2i(10, 10), "form helper");
  gui->add_group("basic types");
  gui->add_variable("bvar", bvar);
  gui->add_variable("ivar", ivar);
  gui->add_variable("dvar", dvar);
  gui->add_variable("strval", strval);
  gui->add_group("other widgets");
  gui->add_button("button", []() {
    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("button") << (" ") << (std::endl) << (std::flush);
  });
  screen->set_visible(true);
  screen->perform_layout();
  window->center();
  nanogui::mainloop(-1);
  nanogui::shutdown();
  return 0;
}