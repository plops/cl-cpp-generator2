// david kirk: programming massively parallel processors (third ed) p. 175
// prefix sum
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
auto g_start = static_cast<typeof(
    std::chrono::high_resolution_clock::now().time_since_epoch().count())>(0);
void sequential_scan(float *x, float *y, int n) {
  auto accum = x[0];
  y[0] = accum;
  for (int i = 1; i < n; (i)++) {
    (accum) += (x[i]);
    y[i] = accum;
  };
}
int main() {
  g_start =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  auto n_cuda = 0;
  {
    auto r = cudaGetDeviceCount(&n_cuda);
    (std::cout) << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (g_start)))
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
                     (g_start)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cudaSetDevice(0) => ") << (r) << (" '")
                << (cudaGetErrorString(r)) << ("' ") << (std::endl);
    assert((cudaSuccess) == (r));
  };
  return 0;
}