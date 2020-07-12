
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

#include "arpack-ng/ICB/arpack.hpp"

using namespace std::chrono_literals;
State state = {};
__global__ void kernel_hamiltonian(float *out, float *in) {
  auto idx = ((((blockIdx.x) * (blockDim.x))) + (threadIdx.x));
  if ((idx) < (1000)) {
    out[idx] = in[idx];
  };
}
int main(int argc, char const *const *const argv) {
  state._main_version = "da1e75818751d00f7a475cb26e0c67ebfdb066e8";
  state._code_repository =
      "https://github.com/plops/cl-cpp-generator2/tree/master/example/19_nvrtc";
  state._code_generation_time = "14:42:17 of Sunday, 2020-07-12 (GMT+1)";
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
  cudaStream_t stream;
  {
    auto res = cudaStreamCreate(&stream);
    if (!((cudaSuccess) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("error:") << (" ") << (std::setw(8))
                  << (" cudaGetErrorString(res)='") << (cudaGetErrorString(res))
                  << ("'") << (std::endl) << (std::flush);
      throw std::runtime_error("cudaStreamCreate(&stream)");
    };
  };
  float *in;
  float *out;
  {
    auto res = cudaMallocManaged(&in, ((1000) * (sizeof(float))));
    if (!((cudaSuccess) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("error:") << (" ") << (std::setw(8))
                  << (" cudaGetErrorString(res)='") << (cudaGetErrorString(res))
                  << ("'") << (std::endl) << (std::flush);
      throw std::runtime_error(
          "cudaMallocManaged(&in, ((1000)*(sizeof(float))))");
    };
  };
  {
    auto res = cudaMallocManaged(&out, ((1000) * (sizeof(float))));
    if (!((cudaSuccess) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("error:") << (" ") << (std::setw(8))
                  << (" cudaGetErrorString(res)='") << (cudaGetErrorString(res))
                  << ("'") << (std::endl) << (std::flush);
      throw std::runtime_error(
          "cudaMallocManaged(&out, ((1000)*(sizeof(float))))");
    };
  };
  kernel_hamiltonian<<<2, 512, 0, stream>>>(out, in);
  cudaStreamSynchronize(stream);
  {
    auto res = cudaFree(out);
    if (!((cudaSuccess) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("error:") << (" ") << (std::setw(8))
                  << (" cudaGetErrorString(res)='") << (cudaGetErrorString(res))
                  << ("'") << (std::endl) << (std::flush);
      throw std::runtime_error("cudaFree(out)");
    };
  };
  {
    auto res = cudaFree(in);
    if (!((cudaSuccess) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("error:") << (" ") << (std::setw(8))
                  << (" cudaGetErrorString(res)='") << (cudaGetErrorString(res))
                  << ("'") << (std::endl) << (std::flush);
      throw std::runtime_error("cudaFree(in)");
    };
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};