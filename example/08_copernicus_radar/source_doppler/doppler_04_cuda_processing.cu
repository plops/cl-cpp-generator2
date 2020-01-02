
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/simpleCUFFT/simpleCUFFT.cu
#include <cassert>

#include "/opt/cuda/targets/x86_64-linux/include/cuda_runtime.h"
#include "/opt/cuda/targets/x86_64-linux/include/cufft.h"
#include "/opt/cuda/targets/x86_64-linux/include/cufftw.h"

typedef float2 Complex;

void initProcessing() {
  auto n_cuda = 0;
  {
    auto r = cudaGetDeviceCount(&n_cuda);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cudaGetDeviceCount(&n_cuda) => ") << (r)
                << (" '") << (cudaGetErrorString(r)) << ("' ") << (" n_cuda=")
                << (n_cuda) << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cudaSetDevice(0);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cudaSetDevice(0) => ") << (r) << (" '")
                << (cudaGetErrorString(r)) << ("' ") << (std::endl);
    assert((cudaSuccess) == (r));
  };
}
void runProcessing(int index) {
  auto p = reinterpret_cast<Complex *>(state._mmap_data);
  auto range = state._range;
  auto h_signal = &(p[((range) * (index))]);
  Complex *d_signal;
  Complex *d_kernel;
  auto memsize = ((sizeof(Complex)) * (range));
  {
    auto r = cudaMalloc(reinterpret_cast<void **>(&d_signal), memsize);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (state._start_time)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" cudaMalloc(reinterpret_cast<void**>(&d_signal), memsize) => ")
        << (r) << (" '") << (cudaGetErrorString(r)) << ("' ") << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cudaMemcpy(d_signal, h_signal, memsize, cudaMemcpyHostToDevice);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__)
                << (" cudaMemcpy(d_signal, h_signal, memsize, "
                    "cudaMemcpyHostToDevice) => ")
                << (r) << (" '") << (cudaGetErrorString(r)) << ("' ")
                << (std::endl);
    assert((cudaSuccess) == (r));
  };
  cufftHandle plan;
  {
    auto r = cufftPlan1d(&plan, range, CUFFT_C2C, 1);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__)
                << (" cufftPlan1d(&plan, range, CUFFT_C2C, 1) => ") << (r)
                << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (state._start_time)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD) => ") << (r)
        << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cudaMalloc(reinterpret_cast<void **>(&d_kernel),
                        ((sizeof(Complex)) * (range)));
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__)
                << (" cudaMalloc(reinterpret_cast<void**>(&d_kernel), "
                    "((sizeof(Complex))*(range))) => ")
                << (r) << (" '") << (cudaGetErrorString(r)) << ("' ")
                << (std::endl);
    assert((cudaSuccess) == (r));
  };
}
void cleanupProcessing(){};