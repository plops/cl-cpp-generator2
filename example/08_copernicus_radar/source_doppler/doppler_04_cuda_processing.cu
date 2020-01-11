
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

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = ((((a.x) * (b.x))) - (((a.y) * (b.y))));
  c.y = ((((a.x) * (b.y))) + (((a.y) * (b.x))));
  return c;
}
static __global__ void ComplexPointwiseMul(Complex *a, Complex *b, int size) {
  auto numThreads = ((blockDim.x) * (gridDim.x));
  auto threadID = ((((blockIdx.x) * (blockDim.x))) + (threadIdx.x));
  for (int i(threadID); i < size; (i) += (numThreads)) {
    a[i] = ComplexMul(a[i], b[i]);
  };
}
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
std::complex<float> *runProcessing(int index) {
  auto p = reinterpret_cast<Complex *>(state._mmap_data);
  auto range = state._range;
  auto h_signal = &(p[((range) * (index))]);
  Complex *d_signal;
  Complex *d_kernel;
  auto memsize = ((sizeof(Complex)) * (range));
  static Complex *h_signal2 = static_cast<Complex *>(malloc(memsize));
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
  ComplexPointwiseMul<<<32, 256>>>(d_signal, d_kernel, range);
  {
    auto r = cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (state._start_time)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE) => ") << (r)
        << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cudaMemcpy(h_signal2, d_signal, memsize, cudaMemcpyDeviceToHost);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__)
                << (" cudaMemcpy(h_signal2, d_signal, memsize, "
                    "cudaMemcpyDeviceToHost) => ")
                << (r) << (" '") << (cudaGetErrorString(r)) << ("' ")
                << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cufftDestroy(plan);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cufftDestroy(plan) => ") << (r)
                << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cudaFree(d_signal);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cudaFree(d_signal) => ") << (r) << (" '")
                << (cudaGetErrorString(r)) << ("' ") << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cudaFree(d_kernel);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cudaFree(d_kernel) => ") << (r) << (" '")
                << (cudaGetErrorString(r)) << ("' ") << (std::endl);
    assert((cudaSuccess) == (r));
  };
  return reinterpret_cast<std::complex<float> *>(h_signal2);
}
void cleanupProcessing(){};