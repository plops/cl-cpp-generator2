#include "DCGANGeneratorImpl.h"
#include <iostream>
#include <spdlog/spdlog.h>
TORCH_MODULE(DCGANGenerator);
int main(int argc, char **argv) {
  spdlog::info("start  argc='{}'", argc);
  torch::Tensor tensor = torch::eye(3);
  (std::cout) << (tensor) << (std::endl);
  auto kNoiseSize = 12;
  auto generator = DCGANGenerator(kNoiseSize);
  auto discriminator = torch::nn::Sequential(
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
      torch::nn::LeakyReLU(
          torch::nn::LeakyReLUOptions().negative_slope((0.20f))),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
      torch::nn::BatchNorm2d(128),
      torch::nn::LeakyReLU(
          torch::nn::LeakyReLUOptions().negative_slope((0.20f))),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
      torch::nn::BatchNorm2d(256),
      torch::nn::LeakyReLU(
          torch::nn::LeakyReLUOptions().negative_slope((0.20f))),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
      torch::nn::Sigmoid());
}