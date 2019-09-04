// nvcc -o 04matmul 04matmul.cu
// nvprof 04matmul
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#define SHM_SIZE (16 * 16)

__global__ void matrix_mul(int *a, int *b, int *c, int n) {
  __shared__ int A[SHM_SIZE] = {0};
  __shared__ int B[SHM_SIZE] = {0};
  int col = ((((blockDim.x) * (blockIdx.x))) + (threadIdx.x));
  int row = ((((blockDim.y) * (blockIdx.y))) + (threadIdx.y));
  auto tx = threadIdx.x;
  auto ty = threadIdx.y;
  auto dim = blockDim.x;
  int sum = 0;
  // move tile across length of grid
  for (int i = 0; i < ((((n) + (dim) + (-1))) / (dim)); (i) += (1)) {
    A[((tx) + (((dim) * (ty))))] =
        a[((((i) * (dim))) + (tx) + (((row) * (n))))];
    B[((tx) + (((dim) * (ty))))] =
        b[((((i) * (dim) * (n))) + (((ty) * (n))) + (col))];
    __syncthreads();
    // accumulate partial results
    for (int j = 0; j < dim; (j) += (1)) {
      (sum) +=
          (((A[((((ty) * (dim))) + (j))]) * (B[((((j) * (dim))) + (tx))])));
    }
    __syncthreads();
  }
  c[((col) + (((row) * (n))))] = sum;
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