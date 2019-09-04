#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void vector_add(int *a, int *b, int *c, int n) {
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
int main() {
  auto = 1 << 20;
  auto = ((n) * (sizeof(bytes)));
  int *a;
  int *b;
  int *c;
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);
  init_array(a, n);
  init_array(b, n);
  auto = 256;
  auto = ((((n) + (((threads) - (1))))) / (threads));
  vector_add<<<blocks, threads, 0, 0>>>(a, b, c, n);
}