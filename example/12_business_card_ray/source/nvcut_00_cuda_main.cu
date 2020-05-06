
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
struct v {
  float x;
  float y;
  float z;
  __device__ v operator+(v r) {
    return v(((x) + (r.x)), ((y) + (r.y)), ((z) + (r.z)));
  };
  __device__ v operator*(float r) {
    return v(((x) * (r)), ((y) * (r)), ((z) * (r)));
  };
  __device__ float operator%(v r) {
    return ((((x) * (r.x))) + (((y) * (r.y))) + (((z) * (r.z))));
  };
  __device__ v operator^(v r) {
    return v(((((y) * (r.z))) - (((z) * (r.y)))),
             ((((z) * (r.x))) - (((x) * (r.z)))),
             ((((x) * (r.y))) - (((y) * (r.x)))));
  };
  __device__ v(){};
  __device__ v(float a, float b, float c) {
    x = a;
    y = b;
    z = c;
  };
  __device__ v operator!() {
    return ((*this) * ((((1.0)) / (sqrt(*this % *this)))));
  };
};
__global__ void GetColor(unsigned char *img) {
  auto x = blockIdx.x;
  auto y = threadIdx.x;
  auto cam_dir = !(v(-6, -16, 0));
  auto cam_up = ((!(((v(0, 0, 1)) ^ (cam_dir)))) * ((2.00f - 3)));
};
int main() {
  char *bitmap = new char[((DIM) * (DIM) * (BPP))];
  unsigned char *dev_bitmap;
  cudaMalloc(reinterpret_cast<void **>(&dev_bitmap), ((DIM) * (DIM) * (BPP)));
  GetColor<<<DIM, DIM>>>(dev_bitmap);
  cudaMemcpy(bitmap, dev_bitmap, ((DIM) * (DIM) * (BPP)),
             cudaMemcpyDeviceToHost);
  delete (bitmap);
  return 0;
};