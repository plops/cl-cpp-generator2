
#include "utils.h"

#include "globals.h"

// implementation
;
#include <vis_00_base.hpp>

using namespace std::chrono_literals;

State state = {};
int main(int argc, char **argv) {
  state._main_version = "64e1a9b2748c063f2359bcd826ffdf8b7c0520ef";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/58_stdpar/source/";
  state._code_generation_time = "22:02:52 of Friday, 2020-12-25 (GMT+1)";
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
  return 0;
}