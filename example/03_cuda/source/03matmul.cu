#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;
__global__ void matrix_mul(int *a, int *b, int *c, int n) {
  int tid = ((((blockDim.x) * (blockIdx.x))) + (threadIdx.x));
  if (tid < n) {
    c[tid] = ((a[tid]) + (b[tid]));
  };
}
int main() {
  // 1024x1024 square matrix
  auto n = 1 << 10;
  auto bytes = ((n) * (sizeof(int)));
  int *a;
  int *b;
  int *c;
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);
  init_array(a, n);
  init_array(b, n);
  // no communication between threads, so work splitting not critical
  // add padding
  auto threads = 256;
  auto blocks = ((((n) + (((threads) - (1))))) / (threads));
  // n=1048576 threads=127 blocks=1048702/127=8257
  // n=1048576 threads=128 blocks=1048703/128=8192
  // n=1048576 threads=129 blocks=349568/43=8129
  // n=1048576 threads=200 blocks=41951/8=5243
  // n=1048576 threads=256 blocks=1048831/256=4096
  // n=1048576 threads=257 blocks=1048832/257=4081
  // n=1048576 threads=258 blocks=349611/86=4065
  // async kernel start
  vector_add<<<blocks, threads, 0, 0>>>(a, b, c, n);
  // managed memory need explicit sync
  cudaDeviceSynchronize();
  vector_add_cpu_assert(a, b, c, n);
  return 0;
}