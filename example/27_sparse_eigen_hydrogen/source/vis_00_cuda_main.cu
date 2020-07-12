
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
#include "arpackpp/include/arrssym.h"

using namespace std::chrono_literals;
State state = {};
__global__ void kernel_hamiltonian(float *out, float *in) {
  auto idx = ((((blockIdx.x) * (blockDim.x))) + (threadIdx.x));
  auto ri = ((((1) + (idx))) * ((5.00e-2)));
  auto l = 0;
  auto Z = 1;
  if ((idx) < (1000)) {
    auto Vr = ((((((l) * (((l) + (1))))) / (((ri) * (ri))))) -
               (((((2) * (Z))) / (ri))));
    if ((((1) <= (idx)) && ((idx) <= (998)))) {
      out[idx] = ((((((-1) / ((2.50e-3)))) *
                    (((in[((idx) - (1))]) + (in[((idx) + (1))]))))) +
                  (((((((2) / ((2.50e-3)))) + (Vr))) * (in[idx]))));
    } else {
      if ((idx) == (0)) {
        out[idx] = ((((((-1) / ((2.50e-3)))) * (((in[((idx) + (1))]))))) +
                    (((((((2) / ((2.50e-3)))) + (Vr))) * (in[idx]))));
      } else {
        out[idx] = ((((((-1) / ((2.50e-3)))) * (((in[((idx) - (1))]))))) +
                    (((((((2) / ((2.50e-3)))) + (Vr))) * (in[idx]))));
      }
    };
  };
}
int main(int argc, char const *const *const argv) {
  state._main_version = "6df5a139794d7c0c1f6014dc0c3d1ccea7e2c723";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/27_sparse_eigen_hydrogen";
  state._code_generation_time = "15:56:06 of Sunday, 2020-07-12 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start main") << (" ")
      << (std::setw(8)) << (" state._main_version='") << (state._main_version)
      << ("'") << (std::endl) << (std::flush);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
      << (std::setw(8)) << (" state._code_repository='")
      << (state._code_repository) << ("'") << (std::endl) << (std::flush);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
      << (std::setw(8)) << (" state._code_generation_time='")
      << (state._code_generation_time) << ("'") << (std::endl) << (std::flush);
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
  // relevant arpack++ example
  // https://github.com/m-reuter/arpackpp/blob/master/examples/reverse/sym/rsymreg.cc
  ;
  // The following values of which are available:
  // which = 'LM' : Eigenvalues with largest magnitude (eigs, eigsh), that is,
  // largest eigenvalues in the euclidean norm of complex numbers. which = 'SM'
  // : Eigenvalues with smallest magnitude (eigs, eigsh), that is, smallest
  // eigenvalues in the euclidean norm of complex numbers. which = 'LR' :
  // Eigenvalues with largest real part (eigs). which = 'SR' : Eigenvalues with
  // smallest real part (eigs). which = 'LI' : Eigenvalues with largest
  // imaginary part (eigs). which = 'SI' : Eigenvalues with smallest imaginary
  // part (eigs). which = 'LA' : Eigenvalues with largest algebraic value
  // (eigsh), that is, largest eigenvalues inclusive of any negative sign. which
  // = 'SA' : Eigenvalues with smallest algebraic value (eigsh), that is,
  // smallest eigenvalues inclusive of any negative sign. which = 'BE' :
  // Eigenvalues from both ends of the spectrum (eigsh). Note that ARPACK is
  // generally better at finding extremal eigenvalues, that is, eigenvalues with
  // large magnitudes. In particular, using which = 'SM' may lead to slow
  // execution time and/or anomalous results. A better approach is to use
  // shift-invert mode.
  ;
  auto prob = ARrcSymStdEig<float>(1000, 8L, "BE", 0, (1.00e-4), 100000);
  while (!(prob.ArnoldiBasisFound())) {
    prob.TakeStep();
    auto ido = prob.GetIdo();
    if ((((ido) == (1)) || ((ido) == (-1)))) {
      auto in_ = prob.GetVector();
      auto out_ = prob.PutVector();
      // multiply
      for (auto i = 0; (i) < (1000); (i) += (1)) {
        auto v = in_[i];
        in[i] = v;
      }
      kernel_hamiltonian<<<2, 512, 0, stream>>>(out, in);
      cudaStreamSynchronize(stream);
      for (auto i = 0; (i) < (1000); (i) += (1)) {
        auto v = out[i];
        out_[i] = v;
      };
    };
  }
  prob.FindEigenvectors();
  for (auto i = 0; (i) < (8); (i) += (1)) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" prob.Eigenvalue(i)='")
                << (prob.Eigenvalue(i)) << ("'") << (std::endl) << (std::flush);
  };
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