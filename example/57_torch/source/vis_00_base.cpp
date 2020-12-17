
#include "utils.h"

#include "globals.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <torch/torch.h>

using namespace std::chrono_literals;

State state = {};
int main(int argc, char **argv) {
  state._main_version = "3d621ed6ff7e3e6f30b7dcfebe57563a8d498d09";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/57_torch/source/";
  state._code_generation_time = "20:19:00 of Thursday, 2020-12-17 (GMT+1)";
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
  auto tensor = torch::eye(3);
  {

    auto lock = std::unique_lock<std::mutex>(state._stdout_mutex);
    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" tensor='") << (tensor) << ("'")
                << (std::endl) << (std::flush);
  }
  return 0;
}