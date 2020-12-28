
#include "utils.h"

#include "globals.h"

// implementation
;
#include <vis_00_base.hpp>

using namespace std::chrono_literals;

State state = {};
int jacobi_solver(float *data, int M, int N, float max_diff) {
  // c++: 5min49s
  // https://on-demand.gputechconf.com/supercomputing/2019/video/sc1936-gpu-programming-with-standard-c++17/

  // python:
  // https://developer.nvidia.com/blog/accelerating-python-on-gpus-with-nvc-and-cython/
  ;
  auto temp = std::make_unique<float[]>(((M) * (N)));
  std::copy(std::execution::par, data, ((data) + (((M) * (N)))), temp.get());
  auto iterations = 0;
  auto keep_going = true;
  auto from = data;
  auto to = temp.get();
  while (keep_going) {
    (iterations)++;
    std::for_each(std::execution::par);
  }
}
int main(int argc, char **argv) {
  state._main_version = "0a6d69c725e26d7b8882fb5cf558ffad9ce19649";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/58_stdpar/source/";
  state._code_generation_time = "13:06:29 of Monday, 2020-12-28 (GMT+1)";
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