
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

#include "data.h"

#include <complex>

#include <cmath>

// fixme: tx configuration for each pulse is currently always the same. for iw
// datasets i have to figure out how to get the tx configuration rank packets in
// the past.

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
  Complex *p = reinterpret_cast<Complex *>(state._mmap_data);
  auto range = state._range;
  Complex *h_signal = &(p[((range) * (index))]);
  Complex *d_signal;
  Complex *d_signal_out;
  Complex *d_kernel;
  Complex *d_kernel_out;
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
        << (r) << (" '") << (cudaGetErrorString(r)) << ("' ") << (" memsize=")
        << (memsize) << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cudaMalloc(reinterpret_cast<void **>(&d_signal_out), memsize);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (state._start_time)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" cudaMalloc(reinterpret_cast<void**>(&d_signal_out), memsize) => ")
        << (r) << (" '") << (cudaGetErrorString(r)) << ("' ") << (" memsize=")
        << (memsize) << (std::endl);
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
                << (" memsize=") << (memsize) << (" d_signal=") << (d_signal)
                << (" h_signal=") << (h_signal) << (" p=") << (p)
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
                << (" range=") << (range) << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cufftExecC2C(plan, d_signal, d_signal_out, CUFFT_FORWARD);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (state._start_time)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" cufftExecC2C(plan, d_signal, d_signal_out, CUFFT_FORWARD) => ")
        << (r) << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cudaMalloc(reinterpret_cast<void **>(&d_kernel), memsize);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (state._start_time)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" cudaMalloc(reinterpret_cast<void**>(&d_kernel), memsize) => ")
        << (r) << (" '") << (cudaGetErrorString(r)) << ("' ") << (" memsize=")
        << (memsize) << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cudaMalloc(reinterpret_cast<void **>(&d_kernel_out), memsize);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (state._start_time)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" cudaMalloc(reinterpret_cast<void**>(&d_kernel_out), memsize) => ")
        << (r) << (" '") << (cudaGetErrorString(r)) << ("' ") << (" memsize=")
        << (memsize) << (std::endl);
    assert((cudaSuccess) == (r));
  };
  if (!(state._kernel_arg)) {
    state._kernel_arg =
        static_cast<float *>(malloc(((sizeof(float)) * (state._range))));
  };
  auto _txprr = txprr[index];
  auto _txpl = txpl[index];
  auto _txpsf = txpsf[index];
  auto _fdec = fdec[index];
  auto h_kernel = static_cast<std::complex<float> *>(malloc(memsize));
  auto xs_off = 0;
  for (int i(0); xs_off < _txpl; (i)++) {
    const std::complex<float> imag(0, 1);
    auto xs_us = ((i) / (_fdec));
    xs_off = ((xs_us) - ((((5.e-1f)) * (_txpl))) - ((5.e-1f)));
    auto arg =
        ((2) * ((3.1415927e+0f)) *
         (((((xs_off) * (((_txpsf) + ((((5.e-1f)) * (_txpl) * (_txprr))))))) +
           (((((xs_off) * (xs_off))) * ((5.e-1f)) * (_txprr))))));
    auto cplx = std::exp(((imag) * (arg)));
    h_kernel[i] = cplx;
    state._kernel_arg[i] = arg;
  }
  {
    auto r = cudaMemcpy(d_kernel, h_kernel, memsize, cudaMemcpyHostToDevice);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__)
                << (" cudaMemcpy(d_kernel, h_kernel, memsize, "
                    "cudaMemcpyHostToDevice) => ")
                << (r) << (" '") << (cudaGetErrorString(r)) << ("' ")
                << (" memsize=") << (memsize) << (std::endl);
    assert((cudaSuccess) == (r));
  };
  {
    auto r = cufftExecC2C(plan, d_kernel, d_kernel_out, CUFFT_FORWARD);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (state._start_time)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" cufftExecC2C(plan, d_kernel, d_kernel_out, CUFFT_FORWARD) => ")
        << (r) << (std::endl);
    assert((cudaSuccess) == (r));
  };
  free(h_kernel);
  ComplexPointwiseMul<<<128, 1024>>>(d_signal_out, d_kernel_out, range);
  // copy data back
  {
    auto h_signal3 = static_cast<Complex *>(malloc(memsize));
    auto v = reinterpret_cast<std::complex<float> *>(h_signal3);
    {
      auto r =
          cudaMemcpy(h_signal3, d_signal_out, memsize, cudaMemcpyDeviceToHost);
      (std::cout) << (((std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count()) -
                       (state._start_time)))
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__)
                  << (" cudaMemcpy(h_signal3, d_signal_out, memsize, "
                      "cudaMemcpyDeviceToHost) => ")
                  << (r) << (" '") << (cudaGetErrorString(r)) << ("' ")
                  << (" memsize=") << (memsize) << (std::endl);
      assert((cudaSuccess) == (r));
    };
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("runProcessing") << (" ")
                << (std::setw(8)) << (" v[0]=") << (v[0]) << (std::setw(8))
                << (" v[1]=") << (v[1]) << (std::setw(8)) << (" v[2]=")
                << (v[2]) << (std::setw(8)) << (" v[3]=") << (v[3])
                << (std::setw(8)) << (" v[4]=") << (v[4]) << (std::endl);
    free(h_signal3);
  };
  {
    auto r = cufftExecC2C(plan, d_signal_out, d_signal, CUFFT_INVERSE);
    (std::cout)
        << (((std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()) -
             (state._start_time)))
        << (" ") << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
        << (" cufftExecC2C(plan, d_signal_out, d_signal, CUFFT_INVERSE) => ")
        << (r) << (std::endl);
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
                << (" memsize=") << (memsize) << (std::endl);
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
    auto r = cudaFree(d_signal_out);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cudaFree(d_signal_out) => ") << (r)
                << (" '") << (cudaGetErrorString(r)) << ("' ") << (std::endl);
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
  {
    auto r = cudaFree(d_kernel_out);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cudaFree(d_kernel_out) => ") << (r)
                << (" '") << (cudaGetErrorString(r)) << ("' ") << (std::endl);
    assert((cudaSuccess) == (r));
  };
  return reinterpret_cast<std::complex<float> *>(h_signal2);
}
void cleanupProcessing(){};