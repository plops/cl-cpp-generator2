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
  "// n=1048576 threads=127 blocks=1048702/127=8257";
  "// n=1048576 threads=128 blocks=1048703/128=8192";
  "// n=1048576 threads=129 blocks=349568/43=8129";
  "// n=1048576 threads=200 blocks=41951/8=5243";
  "// n=1048576 threads=256 blocks=1048831/256=4096";
  "// n=1048576 threads=257 blocks=1048832/257=4081";
  "// n=1048576 threads=258 blocks=349611/86=4065";
}