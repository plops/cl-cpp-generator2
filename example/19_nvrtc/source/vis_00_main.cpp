
#include "utils.h"

#include "globals.h"

;
// g++ -march=native -Ofast --std=gnu++20 vis_00_main.cpp
// -I/media/sdb4/cuda/11.0.1/include/ -L /media/sdb4/cuda/11.0.1/lib -lcudart
// -lcuda
#include "vis_01_rtc.cpp"
#include "vis_02_cu_device.cpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
using namespace std::chrono_literals;
State state = {};
int main() {
  state._main_version = "8a5bb1b50447feb1c3902c7b99fecb52eb782ecd";
  state._code_repository =
      "https://github.com/plops/cl-cpp-generator2/tree/master/example/19_nvrtc";
  state._code_generation_time = "10:43:14 of Sunday, 2020-06-21 (GMT+1)";
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
  auto dev = CudaDevice::FindByProperties(
      CudaDeviceProperties::ByIntegratedType(false));
  dev.setAsCurrent();
  auto ctx = CudaContext(dev);
  auto code = Code::FromFile("bla.cu");
  auto program = Program("myprog", code);
  auto kernel =
      Kernel("setKernel").instantiate<float, std::integral_constant<int, 10>>();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};