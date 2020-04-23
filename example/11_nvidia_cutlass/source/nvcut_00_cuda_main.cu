
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
/*
  export
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/martin/src/cutlass/build/tools/library/
  export PATH=$PATH:/opt/cuda/nvvm/bin/
  /opt/cuda/bin/nvcc nvcut_00_cuda_main.cu  -I /home/martin/src/cutlass/include/
  -I /opt/cuda/include/ -I/home/martin/src/cutlass/tools/util/include/
  -I/home/martin/src/tools/library/include
  -L/home/martin/src/cutlass/build/tools/library/ -lcutlass --std=c++14 -O1 -g
  -Xcompiler=-march=native
  --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0
*/
// https://github.com/NVIDIA/cutlass/blob/master/media/docs/quickstart.md
// https://github.com/NVIDIA/cutlass/blob/master/media/docs/functionality.md
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

State state = {};
using namespace std::chrono_literals;
int main() {
  using Gemm =
      cutlass::gemm::device::Gemm<cutlass::half_t, cutlass::layout::ColumnMajor,
                                  cutlass::half_t, cutlass::layout::ColumnMajor,
                                  cutlass::half_t, cutlass::layout::ColumnMajor,
                                  float, cutlass::arch::OpClassTensorOp,
                                  cutlass::arch::Sm75>;
  Gemm gemm_op;
  cutlass::Status status;
  auto M = 512;
  auto N = 256;
  auto K = 128;
  auto alpha = float((1.250));
  auto beta = float((-1.250));
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});
  auto *ptrA = static_cast<cutlass::half_t const *>(A.device_data());
  auto *ptrB = static_cast<cutlass::half_t const *>(B.device_data());
  auto *ptrC = static_cast<cutlass::half_t const *>(C.device_data());
  auto *ptrD = static_cast<cutlass::half_t *>(C.device_data());
  auto ldA = A.device_ref().stride(0);
  auto ldB = B.device_ref().stride(0);
  auto ldC = C.device_ref().stride(0);
  auto ldD = C.device_ref().stride(0);
  status = gemm_op({{M, N, K},
                    {ptrA, ldA},
                    {ptrB, ldB},
                    {ptrC, ldC},
                    {ptrD, ldD},
                    {alpha, beta}});
  if (!((status) == (cutlass::Status::kSuccess))) {
    return -1;
  };
  return 0;
};