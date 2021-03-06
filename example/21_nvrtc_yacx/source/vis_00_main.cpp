
#include "utils.h"

#include "globals.h"

;
// g++ -march=native -Ofast --std=gnu++20 vis_00_main.cpp
// -I/media/sdb4/cuda/11.0.1/include/ -L /media/sdb4/cuda/11.0.1/lib -lcudart
// -lcuda
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
#include <nvrtc.h>

#include <yacx/main.hpp>

using namespace std::chrono_literals;
State state = {};
int main(int argc, char const *const *const argv) {
  state._main_version = "04524a06715c1ac7c174e5edd6458b3c0afbd3d5";
  state._code_repository =
      "https://github.com/plops/cl-cpp-generator2/tree/master/example/19_nvrtc";
  state._code_generation_time = "20:37:33 of Saturday, 2020-06-27 (GMT+1)";
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
  try {
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
    const int size = 32;
    const int data = 1;
    const int times = 4;
    auto v = std::vector<int>();
    static_assert((0) == (size % times));
    v.resize(size);
    std::fill(v.begin(), v.end(), 0);
    auto args = std::vector<yacx::KernelArg>(
        {yacx::KernelArg{v.data(), ((sizeof(int)) * (v.size())), true},
         yacx::KernelArg{const_cast<int *>(&data)}});
    auto grid = dim3(((v.size()) / (times)));
    auto block = dim3(1);
    source.program("my_kernel")
        .instantiate(yacx::type_of(data), times)
        .compile(options)
        .configure(grid, block)
        .launch(args, device);
    std::copy(v.begin(), v.end(),
              std::experimental::make_ostream_joiner(std::cout, ","));
  } catch (const std::exception &e) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("error") << (" ") << (std::setw(8)) << (" e.what()='")
                << (e.what()) << ("'") << (std::endl) << (std::flush);
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};