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
class DCGANGeneratorImpl : public torch::nn::Module {
public:
  DCGANGeneratorImpl(int k_noise_size);
  torch::Tensor forward(torch::Tensor x);
  torch::nn::ConvTranspose2d conv1;
  torch::nn::BatchNorm2d batch_norm1;
  torch::nn::ConvTranspose2d conv2;
  torch::nn::BatchNorm2d batch_norm2;
  torch::nn::ConvTranspose2d conv3;
  torch::nn::BatchNorm2d batch_norm3;
  torch::nn::ConvTranspose2d conv4;
};
int main(int argc, char **argv);
#endif