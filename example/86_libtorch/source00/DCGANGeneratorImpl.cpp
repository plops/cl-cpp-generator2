// no preamble
#include "DCGANGeneratorImpl.h"
#include <spdlog/spdlog.h>
#include <torch/autograd.h>
#include <torch/cuda.h>
DCGANGeneratorImpl::DCGANGeneratorImpl(int kNoiseSize)
    : conv1(torch::nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)),
      conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3)
                .stride(2)
                .padding(1)
                .bias(false)),
      conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4)
                .stride(2)
                .padding(1)
                .bias(false)),
      conv4(
          torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(
              false)),
      batch_norm1(256), batch_norm2(128), batch_norm3(64) {
  spdlog::info("Net constructor");
  register_module("conv1", conv1);
  register_module("conv2", conv2);
  register_module("conv3", conv3);
  register_module("conv4", conv4);
  register_module("batch_norm1", batch_norm1);
  register_module("batch_norm2", batch_norm2);
  register_module("batch_norm3", batch_norm3);
}
torch::Tensor DCGANGeneratorImpl::forward(torch::Tensor x) {
  x = torch::relu(batch_norm1(conv1(x)));
  x = torch::relu(batch_norm2(conv2(x)));
  x = torch::relu(batch_norm3(conv3(x)));
  x = torch::tanh(conv4(x));
  return x;
}