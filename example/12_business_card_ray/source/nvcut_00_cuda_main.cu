
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
// /opt/cuda/bin/nvcc nvcut_00_cuda_main.cu --gpu-architecture=compute_75
// --gpu-code=compute_75 --use_fast_math  -I/opt/cuda/include/ --std=c++14 -O3
// -g -Xcompiler=-march=native
// --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0
#include <cstdio>
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
    return ((*this) * ((((1.0f)) / (sqrtf(*this % *this)))));
  };
};
__device__ int g_seed = 1;
__device__ float R() {
  g_seed = ((((214013) * (g_seed))) + (2531011));
  return (((((g_seed) >> (16)) & (0x7fff))) / ((66635.f)));
};
__device__ int G[9] = {247570, 280596, 280600, 249748, 18578,
                       18577,  231184, 16,     16};
__device__ int TraceRay(v origin, v destination, float &tau, v &normal) {
  tau = (1.0e+9);
  auto m = 0;
  auto p = ((-origin.z) / (destination.z));
  if ((1.00e-2) < p) {
    tau = p;
    normal = v(0, 0, 1);
    m = 1;
  };
  for (int k = 19; 0 < k; k--) {
    for (int j = 9; 0 < j; j--) {
      if (((G[j]) & ((1) << (k)))) {
        auto p = ((origin) + (v(-k, 0, ((-j) - (4)))));
        auto b = p % destination;
        auto c = ((p % p) - ((1.0f)));
        auto q = ((((b) * (b))) - (c));
        if (0 < q) {
          auto s = ((-b) - (sqrtf(q)));
          if (((s < tau) && ((1.00e-2) < s))) {
            tau = s;
            normal = !(((p) + (((destination) * (tau)))));
            m = 2;
          };
        };
      };
    }
  }
  return m;
};
__device__ v Sample(v origin, v destination, int r) {
  auto tau = (0.f);
  auto normal = v();
  if (4 < r) {
    return v(0, 80, 0);
  };
  auto match = TraceRay(origin, destination, tau, normal);
  if (!(match)) {
    return v(0, 0, 80);
  };
  auto intersection = ((origin) + (((destination) * (tau))));
  auto light_dir =
      ((!(v(((9) + (R())), ((9) + (R())), 16))) + (((intersection) * (-1))));
  auto half_vec =
      ((destination) + (((normal) * (((normal % destination) * (-2))))));
  auto lamb_f = light_dir % normal;
  if (((lamb_f < 0) || (TraceRay(intersection, light_dir, tau, normal)))) {
    lamb_f = 0;
  };
  if (((match) & (1))) {
    intersection = ((intersection) * ((0.20f)));
    auto c = (((static_cast<int>(
                   ((ceilf(intersection.x)) + (ceilf(intersection.y))))) &
               (1)))
                 ? (v(3, 1, 1))
                 : (v(3, 3, 3));
    return ((c) * (((((lamb_f) * ((0.20f)))) + ((0.10f)))));
  };
  float color = powf(((light_dir % half_vec) * (1)), (99.f));
  return ((v(color, color, color)) +
          (((Sample(intersection, half_vec, ((r) + (1)))) * ((0.50f)))));
};
__global__ void GetColor(unsigned char *img) {
  auto x = blockIdx.x;
  auto y = threadIdx.x;
  auto cam_dir = !(v((-2.0f), (-16.f), (0.f)));
  auto s = (4.00e-3);
  auto cam_up = ((!(((v((0.f), (0.f), (1.0f))) ^ (cam_dir)))) * (s));
  auto cam_right = ((!(((cam_dir) ^ (cam_up)))) * (s));
  auto eye_offset = ((((((cam_up) + (cam_right))) * (-256))) + (cam_dir));
  auto color = v((13.f), (13.f), (13.f));
  for (int r = 0; r < 64; (r) += (1)) {
    auto delta = ((((cam_up) * (99))) + (((cam_right) * (99))));
    color = ((((Sample(((v(17, 16, 8)) + (delta)),
                       !(((((((delta) * (-1))) + (((cam_up) * (x))) +
                            (((cam_right) * (y))) + (eye_offset))) *
                          (16))),
                       0)) *
               ((3.50f)))) +
             (color));
  }
  img[((((DIM) * (y) * (BPP))) + (((BPP) * (x))) + (0))] = color.x;
  img[((((DIM) * (y) * (BPP))) + (((BPP) * (x))) + (1))] = color.y;
  img[((((DIM) * (y) * (BPP))) + (((BPP) * (x))) + (2))] = color.z;
};
int main() {
  char *bitmap = new char[((DIM) * (DIM) * (BPP))];
  unsigned char *dev_bitmap;
  cudaMalloc(reinterpret_cast<void **>(&dev_bitmap), ((DIM) * (DIM) * (BPP)));
  GetColor<<<DIM, DIM>>>(dev_bitmap);
  cudaMemcpy(bitmap, dev_bitmap, ((DIM) * (DIM) * (BPP)),
             cudaMemcpyDeviceToHost);
  printf("P6 512 512 255 ");
  auto c = bitmap;
  for (int y = DIM; y; y--) {
    for (int x = DIM; x; x--) {
      c = &(bitmap[((((y) * (DIM) * (BPP))) + (((x) * (BPP))))]);
      printf("%c%c%c", c[0], c[1], c[2]);
      (c) += (BPP);
    }
  };
  delete (bitmap);
  return 0;
};