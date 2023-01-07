#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
struct Pixel {
  unsigned char red;
};
typedef struct Pixel Pixel;
int main(int argc, char **argv) {
  (void)argv;
  spdlog::info("start  argc='{}'", argc);
}