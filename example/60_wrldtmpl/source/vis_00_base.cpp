
#include "utils.h"

#include "globals.h"

// implementation

#include <vis_00_base.hpp>

using namespace std::chrono_literals;

State state{{}};

int main(int argc, char **argv) {
  (state._main_version) = ("1f181d44803029cf3c294696d9cb8dc63088b84e");
  (state._code_repository) = ("https://github.com/plops/cl-cpp-generator2/tree/"
                              "master/example/60_wrldtmpl/source/");
  (state._code_generation_time) = ("12:06:19 of Thursday, 2024-05-09 (GMT+1)");
  (state._start_time) =
      (std::chrono::high_resolution_clock::now().time_since_epoch().count());
  {

    auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
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
