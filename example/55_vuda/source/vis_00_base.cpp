
#include "utils.h"

#include "globals.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <vuda_runtime.hpp>

using namespace std::chrono_literals;

State state = {};
void run_vuda() { cudaSetDevice(0); }
int main(int argc, char **argv) {
  state._main_version = "19d1a108aefa1bbc00679cb08f49000d89dbfcb7";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/55_vuda/source/";
  state._code_generation_time = "23:48:02 of Friday, 2020-12-11 (GMT+1)";
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
  run_vuda();
  return 0;
}