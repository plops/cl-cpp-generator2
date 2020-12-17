
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
dcgan_generatorImpl::dcgan_generatorImpl(int k_noise_size)
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
      batch_norm3(128),
      conv4(torch::nn::ConvTranspose2dOptions(c64, 1, 4).bias(false)),
      batch_norm4(64) {
  register_module("conv1", conv1);
  register_module("batch_norm1", batch_norm1);
  register_module("conv2", conv2);
  register_module("batch_norm2", batch_norm2);
  register_module("conv3", conv3);
  register_module("batch_norm3", batch_norm3);
  register_module("conv4", conv4);
  register_module("batch_norm4", batch_norm4);
}
torch::Tensor dcgan_generatorImpl::forward(torch::Tensor x) {
  x = torch::relu(batch_norm1(conv1(x)));
  x = torch::relu(batch_norm2(conv2(x)));
  x = torch::relu(batch_norm3(conv3(x)));
  x = torch::tanh(conv4(x));
  return x;
}
TORCH_MODULE(dcgan_generator);
int main(int argc, char **argv) {
  state._main_version = "fe43becfaf7cf90eb2bd305bc6b1d2d467344e68";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/57_torch/source/";
  state._code_generation_time = "23:35:33 of Thursday, 2020-12-17 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  {

    auto lock = std::unique_lock<std::mutex>(state._stdout_mutex);
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
  auto tensor = torch::eye(3);
  {

    auto lock = std::unique_lock<std::mutex>(state._stdout_mutex);
    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" tensor='") << (tensor) << ("'")
                << (std::endl) << (std::flush);
  }
  return 0;
}