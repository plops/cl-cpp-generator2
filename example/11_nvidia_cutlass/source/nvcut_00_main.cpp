
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
// nvcc nvcut_00_main.cpp  -I /home/martin/src/cutlass/include/ -I
// /opt/cuda/include/ -I/home/martin/src/cutlass/tools/util/include/ --std=c++14
// -O1 -g -Xcompiler=-march=native
// --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

State state = {};
using namespace std::chrono_literals;
int main() {
  using Gemm =
      cutlass::gemm::device::Gemm<
          cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t, cutlass::layout::ColumnMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75, > >
      ;
  Gemm gemm_op;
  cutlass::Status status;
  return 0;
};