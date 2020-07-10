
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
void init_input(float *a, size_t size) {
  for (auto i = 0; (i) < (size); (i) += (1)) {
    a[i] = (((1.0f)) * (i));
  }
}
int main(int argc, char const *const *const argv) {
  state._main_version = "91dc349f8a73dfc90c5ad7bfa27a2f3f898039ba";
  state._code_repository =
      "https://github.com/plops/cl-cpp-generator2/tree/master/example/19_nvrtc";
  state._code_generation_time = "21:32:46 of Monday, 2020-07-06 (GMT+1)";
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
  auto blocks = 512;
  auto threads = 512;
  if (!((cudaSuccess) == (cudaStreamCreate(&stream)))) {
    throw std::runtime_error("cudaStreamCreate(&stream)");
  };
  float *in;
  float *out;
  if (!((cudaSuccess) == (cudaMallocManaged(&in, ((N) * (sizeof(float))))))) {
    throw std::runtime_error("cudaMallocManaged(&in, ((N)*(sizeof(float))))");
  };
  if (!((cudaSuccess) == (cudaMallocManaged(&out, ((N) * (sizeof(float))))))) {
    throw std::runtime_error("cudaMallocManaged(&out, ((N)*(sizeof(float))))");
  };
  init_input(in, N);
  auto graph_created = false;
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  for (auto istep = 0; (istep) < (NSTEP); (istep) += (1)) {
    if (!(graph_created)) {
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
      for (auto ik = 0; (ik) < (NKERNEL); (ik) += (1)) {
        shortKernel<<<blocks, threads, 0, stream>>>(out, in);
      }
      cudaStreamEndCapture(stream, &graph);
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
      graph_created = true;
    };
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
  };
  if (!((cudaSuccess) == (cudaFree(in)))) {
    throw std::runtime_error("cudaFree(in)");
  };
  if (!((cudaSuccess) == (cudaFree(out)))) {
    throw std::runtime_error("cudaFree(out)");
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};