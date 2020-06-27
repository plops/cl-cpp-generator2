
#include "utils.h"

#include "globals.h"

;
// g++ -march=native -Ofast --std=gnu++20 vis_00_main.cpp
// -I/media/sdb4/cuda/11.0.1/include/ -L /media/sdb4/cuda/11.0.1/lib -lcudart
// -lcuda
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <yacx/main.hpp>

using namespace std::chrono_literals;
State state = {};
int main(int argc, char const *const *const argv) {
  state._main_version = "d18621e9dc0004e30cec464389ce0d245cab7761";
  state._code_repository =
      "https://github.com/plops/cl-cpp-generator2/tree/master/example/19_nvrtc";
  state._code_generation_time = "17:53:55 of Saturday, 2020-06-27 (GMT+1)";
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
  auto device = yacx::Devices::findDevice();
  auto options = yacx::Options(yacx::options::GpuArchitecture(device));
  options.insert("--std", "c++17");
  auto source = yacx::Source(
      R"(template<typename type, int size> __global__ void my_kernel (type* c, type val)    {
            auto idx  = ((((blockIdx.x)*(blockDim.x)))+(threadIdx.x));
    for (auto i = 0;(i)<(size);(i)+=(1)) {
                        c[i]=((idx)+(val));
};
})");

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};