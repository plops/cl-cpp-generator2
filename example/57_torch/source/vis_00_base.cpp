
#include "utils.h"

#include "globals.h"

#include <chrono>
#include <iostream>
#include <thread>

// implementation
;
#include "vis_00_base.hpp"
using namespace std::chrono_literals;

State state = {};
static constexpr int c256 = 256;
static constexpr int c128 = 128;
static constexpr int c64 = 64;
DCGANGeneratorImpl::DCGANGeneratorImpl(int k_noise_size)
    : conv1(
          torch::nn::ConvTranspose2dOptions(k_noise_size, c256, 4).bias(false)),
      batch_norm1(c256), conv2(torch::nn::ConvTranspose2dOptions(c256, c128, 3)
                                   .stride(2)
                                   .padding(1)
                                   .bias(false)),
      batch_norm2(c128), conv3(torch::nn::ConvTranspose2dOptions(c128, c64, 4)
                                   .stride(2)
                                   .padding(1)
                                   .bias(false)),
      batch_norm3(c64),
      conv4(torch::nn::ConvTranspose2dOptions(c64, 1, 4).bias(false)) {
  // k_noise_size is the size of the input noise vector
  ;
  register_module("conv1", conv1);
  register_module("batch_norm1", batch_norm1);
  register_module("conv2", conv2);
  register_module("batch_norm2", batch_norm2);
  register_module("conv3", conv3);
  register_module("batch_norm3", batch_norm3);
  register_module("conv4", conv4);
}
torch::Tensor DCGANGeneratorImpl::forward(torch::Tensor x) {
  x = torch::relu(batch_norm1(conv1(x)));
  x = torch::relu(batch_norm2(conv2(x)));
  x = torch::relu(batch_norm3(conv3(x)));
  x = torch::tanh(conv4(x));
  return x;
}
TORCH_MODULE(DCGANGenerator);
int main(int argc, char **argv) {
  state._main_version = "b23c2ba167247fbc74290e1f05cbb43327f103ba";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/57_torch/source/";
  state._code_generation_time = "00:38:51 of Friday, 2020-12-18 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("start main") << (" ") << (std::setw(8))
                << (" state._main_version='") << (state._main_version) << ("'")
                << (std::setw(8)) << (" state._code_repository='")
                << (state._code_repository) << ("'") << (std::setw(8))
                << (" state._code_generation_time='")
                << (state._code_generation_time) << ("'") << (std::endl)
                << (std::flush);
  }
  torch::manual_seed(1);
  auto device = torch::Device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("we have cuda") << (" ") << (std::setw(8))
                  << (" device='") << (device) << ("'") << (std::endl)
                  << (std::flush);
    }
  }
  const int k_noise_size = 100;
  auto generator = DCGANGenerator(k_noise_size);
  generator->to(device);
  auto discriminator = torch::nn::Sequential(
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(1, c64, 4).stride(2).padding(1).bias(false)),
      torch::nn::LeakyReLU(
          torch::nn::LeakyReLUOptions().negative_slope((0.20f))),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c64, c128, 4)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
      torch::nn::BatchNorm2d(c128),
      torch::nn::LeakyReLU(
          torch::nn::LeakyReLUOptions().negative_slope((0.20f))),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c128, c256, 4)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
      torch::nn::BatchNorm2d(c256),
      torch::nn::LeakyReLU(
          torch::nn::LeakyReLUOptions().negative_slope((0.20f))),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c256, 1, 3)
                            .stride(1)
                            .padding(0)
                            .bias(false)),
      torch::nn::Sigmoid());
  return 0;
}