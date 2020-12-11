
#include "utils.h"

#include "globals.h"

// implementation
;
#include <vis_00_base.hpp>

using namespace std::chrono_literals;

State state = {};
void run_vuda() {
  cudaSetDevice(0);
  const int N = 4096;
  int a[N];
  auto dev_a = static_cast<int *>(nullptr);
  cudaMalloc(reinterpret_cast<void **>(&(dev_a)), ((N) * (sizeof(int))));
  int b[N];
  auto dev_b = static_cast<int *>(nullptr);
  cudaMalloc(reinterpret_cast<void **>(&(dev_b)), ((N) * (sizeof(int))));
  int c[N];
  auto dev_c = static_cast<int *>(nullptr);
  cudaMalloc(reinterpret_cast<void **>(&(dev_c)), ((N) * (sizeof(int))));
  for (auto i = 0; (i) < (N); (i) += (1)) {
    a[i] = ((-1) * (i));
    b[i] = ((i) * (i));
  }
}
int main(int argc, char **argv) {
  state._main_version = "b65b7b1588a74e3385f188e699a986caf105a012";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/55_vuda/source/";
  state._code_generation_time = "00:00:45 of Saturday, 2020-12-12 (GMT+1)";
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