#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void vector_add() {}
void init_array(int *a, int n) {
  for (int i = 0; i < n; (i) += (1)) {
    a[i] = rand() % 100;
  }
}
int main() {
  int n = 1 << 20;
  size_t bytes = ((n) * (sizeof(bytes)));
  int *a;
  int *b;
  int *c;
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);
  init_array(a, n);
  init_array(b, n);
}