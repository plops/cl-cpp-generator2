#include <cassert>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
struct Pixel {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
};
typedef struct Pixel Pixel;
void LoadImage(const std::string &fileName, std::vector<Pixel> &image) {
  auto in = std::ifstream(fileName, std::ios::binary);
  auto header = std::string();
  auto width = int(0);
  auto height = int(0);
  auto maxValue = int(0);
  assert(in.is_open());
  (in) >> (header);
  assert(("P6") == (header));
  (in) >> (width) >> (height) >> (maxValue);
  assert((512) == (width));
  assert((512) == (height));
  assert((255) == (maxValue));
  in.ignore(256, '\n');
  image.reserve(((width) * (height)));
  in.read(reinterpret_cast<char *>(image.data(), ((width) * (height) * (3))));
}
void SaveImage(const std::string &fileName, const std::vector<Pixel> &image) {
  auto out = std::ofstream(fileName, std::ios::binary);
  assert(out.is_open());
  (out) << ("P6\n") << (512 512\n255\n);
  out.write(
      reinterpret_cast<const char *>(image.data(), ((512) * (512) * (3))));
}
__global__ void BoxFilterKernel(const Pixel *input, Pixel restrict *output,
                                int width, int height) {
  int x = ((threadIdx.x) + (((blockIdx.x) * (blockDim.x))));
  int y = ((threadIdx.y) + (((blockIdx.y) * (blockDim.y))));
  int r = 0;
  int g = 0;
  int b = 0;
  if ((width) <= (x)) {
    return;
  }
  if ((height) <= (y)) {
    return;
  }
  for (int i = -2; (i) <= (1); (i)++) {
    for (int j = -2; (j) <= (1); (j)++) {
      int posX = ((x) + (i));
      int posY = ((y) + (j));
      if ((((0) <= (posX)) && ((posX) < (width)) && ((0) <= (posY)) &&
           ((posY) < (height)))) {
        (r) += (input[((posX) + (((width) * (posY))))].red);
        (g) += (input[((posX) + (((width) * (posY))))].green);
        (b) += (input[((posX) + (((width) * (posY))))].blue);
      }
    }
  }
  r = ((r) / (9));
  g = ((g) / (9));
  b = ((b) / (9));
  output[((x) + (((width) * (y))))] = {(unsigned char)r, (unsigned char)g,
                                       (unsigned char)b};
}
int main(int argc, char **argv) {
  (void)argv;
  spdlog::info("start  argc='{}'", argc);
  auto input = std::vector<Pixel>();
  LoadImage("input.ppm", input);
  Pixel *d_in;
  cudaMalloc(&(d_in), ((sizeof(Pixel)) * (512) * (512)));
  Pixel *d_out;
  cudaMalloc(&(d_out), ((sizeof(Pixel)) * (512) * (512)));
  cudaMemcpy(d_in, input.data(), ((sizeof(Pixel)) * (512) * (512)),
             cudaMemcpyHostToDevice);
  auto blockSize = dim3(16, 16);
}