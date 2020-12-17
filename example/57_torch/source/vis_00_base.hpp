#ifndef VIS_00_BASE_H
#define VIS_00_BASE_H
#include "utils.h"
;
#include "globals.h"
;
#include <chrono>
#include <iostream>
#include <thread>
;
// header
;
;
class dcgan_generator_impl : torch::nn::Module {
public:
  dcgan_generator_impl(int k_noise_size);
  torch::nn::ConvTranspose2d conv1;
  torch::nn::BatchNorm2d batch_norm1;
  torch::nn::ConvTranspose2d conv2;
  torch::nn::BatchNorm2d batch_norm2;
  torch::nn::ConvTranspose2d conv3;
  torch::nn::BatchNorm2d batch_norm3;
  torch::nn::ConvTranspose2d conv4;
  torch::nn::BatchNorm2d batch_norm4;
  const int c256 = 256;
  const int c128 = 128;
  const int c64 = 64;
};
int main(int argc, char **argv);
#endif