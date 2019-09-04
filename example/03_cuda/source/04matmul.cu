// nvcc -o 04matmul 04matmul.cu
// nvprof 04matmul
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;
__global__ void matrix_mul(int *a, int *b, int *c, int n) {
  int col = ((((blockDim.x) * (blockIdx.x))) + (threadIdx.x));
  int row = ((((blockDim.y) * (blockIdx.y))) + (threadIdx.y));
  int sum = 0;
  if (((row < n) && (col < n))) {
    for (int k = 0; k < n; (k) += (1)) {
      // row of a times column of b
      (sum) += (((a[((k) + (((row) * (n))))]) * (b[((col) + (((k) * (n))))])));
    }
    c[((col) + (((row) * (n))))] = sum;
  };
}
void matrix_mul_cpu_assert(int *a, int *b, int *c, int n) {
  int tmp = 0;
  // every row i
  for (int i = 0; i < n; (i) += (1)) {
    // every column j
    for (int j = 0; j < n; (j) += (1)) {
      // every row-col pair
      tmp = 0;
      for (int k = 0; k < n; (k) += (1)) {
        (tmp) += (((a[((k) + (((i) * (n))))]) * (b[((j) + (((k) * (n))))])));
      }
      assert((tmp) == (c[((j) + (((i) * (n))))]));
    }
  };
}
void init_matrix(int *a, int n) {
  for (int i = 0; i < ((n) * (n)); (i) += (1)) {
    a[i] = rand() % 100;
  }
}
int main() {
  // 1024x1024 square matrix
  auto n = 1 << 10;
  auto bytes = ((n) * (n) * (sizeof(int)));
  int *a;
  int *b;
  int *c;
  cudaMallocManaged(&(a), bytes);
  cudaMallocManaged(&(b), bytes);
  cudaMallocManaged(&(c), bytes);
  init_matrix(a, n);
  init_matrix(b, n);
  // one thread per output element
  // square thread blocks
  auto threads = 16;
  auto blocks = ((((n) + (((threads) - (1))))) / (threads));
  // n=1024 threads=14 blocks=1037/14=74
  // n=1024 threads=16 blocks=1039/16=64
  // n=1024 threads=32 blocks=1055/32=32
  // kernel launch parameters
  auto threads2 = dim3(threads, threads);
  auto blocks2 = dim3(blocks, blocks);
  // async kernel start
  matrix_mul<<<blocks2, threads2, 0, 0>>>(a, b, c, n);
  // managed memory need explicit sync
  cudaDeviceSynchronize();
  matrix_mul_cpu_assert(a, b, c, n);
  return 0;
}