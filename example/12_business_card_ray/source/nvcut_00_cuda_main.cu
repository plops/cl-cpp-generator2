
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
// /opt/cuda/bin/nvcc nvcut_00_cuda_main.cu  -I/opt/cuda/include/ --std=c++14
// -O1 -g -Xcompiler=-march=native
// --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0
State state = {};
enum { DIM = 512, BPP = 3 };
using namespace std::chrono_literals;
int main() {
  auto bitmap = new char[((DIM) * (DIM) * (BPP))];
  return EXIT_SUCCESS;
};