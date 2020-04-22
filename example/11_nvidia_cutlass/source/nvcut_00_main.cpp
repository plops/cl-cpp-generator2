
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
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
  return 0;
};