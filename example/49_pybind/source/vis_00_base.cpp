
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <pybind11/embed.h>
#include <thread>
using namespace std::chrono_literals;

// implementation
State state = {};
int main(int argc, char **argv) {
  state._main_version = "3c6fedec25c8c6154ab45fa030383a53ec8ea386";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/48_future";
  state._code_generation_time = "21:44:04 of Friday, 2020-12-04 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  {

    auto lock = std::unique_lock<std::mutex>(state._stdout_mutex);
    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("start main") << (" ") << (std::setw(8))
                << (" state._main_version='") << (state._main_version) << ("'")
                << (std::setw(8)) << (" state._code_repository='")
                << (state._code_repository) << ("'") << (std::setw(8))
                << (" state._code_generation_time='")
                << (state._code_generation_time) << ("'") << (std::endl)
                << (std::flush);
  }
  pybind11::exec(R"(print('hello world from PYTHON'))");
  return 0;
}