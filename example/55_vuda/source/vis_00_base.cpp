
#include "utils.h"

#include "globals.h"

// implementation
;
#include <vis_00_base.hpp>

using namespace std::chrono_literals;

State state = {};
void run_vuda() {
  cudaSetDevice(0);
  const int N = 5000;
  const int Nbytes = ((N) * (sizeof(int)));
  int a[N];
  auto dev_a = static_cast<int *>(nullptr);
  cudaMalloc(reinterpret_cast<void **>(&(dev_a)), Nbytes);
  int b[N];
  auto dev_b = static_cast<int *>(nullptr);
  cudaMalloc(reinterpret_cast<void **>(&(dev_b)), Nbytes);
  int c[N];
  auto dev_c = static_cast<int *>(nullptr);
  cudaMalloc(reinterpret_cast<void **>(&(dev_c)), Nbytes);
  for (auto i = 0; (i) < (N); (i) += (1)) {
    a[i] = ((-1) * (i));
    b[i] = ((i) * (i));
  }
  cudaMemcpy(dev_a, a, Nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, Nbytes, cudaMemcpyHostToDevice);
  auto blocks = 128;
  auto threads = 128;
  auto stream_id = 0;
  vuda::launchKernel("/home/martin/src/vuda/samples/simple/add.spv", "main",
                     stream_id, blocks, threads, dev_a, dev_b, dev_c, N);
  cudaMemcpy(c, dev_c, Nbytes, cudaMemcpyDeviceToHost);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}
int main(int argc, char **argv) {
  state._main_version = "7070a947d23a0d2a55ac342b9670d19c8866090b";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/55_vuda/source/";
  state._code_generation_time = "21:52:53 of Friday, 2020-12-25 (GMT+1)";
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