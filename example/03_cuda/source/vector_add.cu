#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std __global__ void vector_add(int *a, int *b, int *c, int n) {
  auto = ((((blockDim.x) * (blockIdx.x))) + (threadIdx.x));
  if (tid < n) {
    c[tid] = ((a[tid]) + (b[tid]));
  };
}
void init_array(int *a, int n) {
  for (int i = 0; i < n; (i) += (1)) {
    a[i] = rand() % 100;
  }
}
void vector_add_cpu_assert(int *a, int *b, int *c, int n) {
  for (int i = 0; i < n; (i) += (1)) {
    assert((c[i]) == (((a[i]) + (b[i]))));
  }
}
int main() {
  auto = 1 << 20;
  auto = ((n) * (sizeof(int)));
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
  auto = 256;
  auto = ((((n) + (((threads) - (1))))) / (threads));
  // n=1048576 threads=127 blocks=1048702/127=8257
  // n=1048576 threads=128 blocks=1048703/128=8192
  // n=1048576 threads=129 blocks=349568/43=8129
  // n=1048576 threads=200 blocks=41951/8=5243
  // n=1048576 threads=256 blocks=1048831/256=4096
  // n=1048576 threads=257 blocks=1048832/257=4081
  // n=1048576 threads=258 blocks=349611/86=4065
  // async kernel start
  vector_add<<<blocks, threads, 0, 0>>>(a, b, c, n);
  cudaDeviceSynchronize();
  vector_add_cpu_assert(a, b, c, n);
}