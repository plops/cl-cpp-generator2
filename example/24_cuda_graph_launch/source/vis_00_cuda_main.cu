
#include "utils.h"

#include "globals.h"

;
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <experimental/iterator>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

enum { N = 500000, NSTEP = 1000, NKERNEL = 20 };
using namespace std::chrono_literals;
State state = {};
__global__ void shortKernel(float *out, float *in) {
  auto idx = ((((blockIdx.x) * (blockDim.x))) + (threadIdx.x));
  if ((idx) < (N)) {
    out[idx] = ((in[idx]) * ((1.230f)));
  };
}
int main(int argc, char const *const *const argv) {
  state._main_version = "5fb4c2cb6e13dfca03ee0dea66562c6c532e8744";
  state._code_repository =
      "https://github.com/plops/cl-cpp-generator2/tree/master/example/19_nvrtc";
  state._code_generation_time = "12:58:38 of Sunday, 2020-07-05 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start main") << (" ")
      << (std::setw(8)) << (" state._main_version='") << (state._main_version)
      << ("'") << (std::setw(8)) << (" state._code_repository='")
      << (state._code_repository) << ("'") << (std::setw(8))
      << (" state._code_generation_time='") << (state._code_generation_time)
      << ("'") << (std::endl) << (std::flush);
  for (auto istep = 0; (istep) < (NSTEP); (istep) += (1)) {
    for (auto ik = 0; (ik) < (NKERNEL); (ik) += (1)) {
      shortKernel<<<blocks, threads, 0, stream>>>(out, in);
      cudaStreamSynchronize(stream);
    }
  }

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};