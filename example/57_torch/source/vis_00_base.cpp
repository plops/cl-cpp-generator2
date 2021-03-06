
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
static constexpr int kNoiseSize = 100;
static constexpr int kBatchSize = 256;
static constexpr int kNumberOfEpochs = 28;
static constexpr int kCheckpointEvery = 200;
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
      batch_norm3(c64), conv4(torch::nn::ConvTranspose2dOptions(c64, 1, 4)
                                  .stride(2)
                                  .padding(1)
                                  .bias(false)) {
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
  state._main_version = "b3d53d871e23eb637a3674fd38d99e84108b7313";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/57_torch/source/";
  state._code_generation_time = "11:53:29 of Saturday, 2020-12-19 (GMT+1)";
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
  auto generator = DCGANGenerator(kNoiseSize);
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
  discriminator->to(device);
  auto dataset =
      torch::data::datasets::MNIST("./mnist")
          .map(torch::data::transforms::Normalize<>((0.50f), (0.50f)))
          .map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(12));
  auto generator_optimizer = torch::optim::Adam(
      generator->parameters(), torch::optim::AdamOptions((2.00e-4f))
                                   .betas(std::make_tuple((0.50f), (0.9990f))));
  auto discriminator_optimizer =
      torch::optim::Adam(discriminator->parameters(),
                         torch::optim::AdamOptions((2.00e-4f))
                             .betas(std::make_tuple((0.50f), (0.9990f))));
  auto checkpoint_counter = 0;
  if (true) {
    torch::load(generator, "generator.pt");
    torch::load(generator_optimizer, "generator_optimizer.pt");
    torch::load(discriminator, "discriminator.pt");
    torch::load(discriminator_optimizer, "discriminator_optimizer.pt");
  }
  for (auto epoch = 0; (epoch) < (kNumberOfEpochs); (epoch) += (1)) {
    auto batch_index = 0;
    for (auto &batch : *data_loader) {
      // train discriminator with real images
      ;
      discriminator->zero_grad();
      auto real_images = batch.data.to(device);
      auto real_labels =
          torch::empty(batch.data.size(0), device).uniform_((0.80f), (1.0f));
      auto real_output = discriminator->forward(real_images);
      auto real_d_loss = torch::binary_cross_entropy(real_output, real_labels);
      real_d_loss.backward();
      // train discriminator with fake images
      ;
      auto noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
      auto fake_images = generator->forward(noise);
      auto fake_labels = torch::zeros(batch.data.size(0), device);
      auto fake_output = discriminator->forward(fake_images.detach());
      auto fake_d_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      fake_d_loss.backward();
      auto d_loss = ((real_d_loss) + (fake_d_loss));
      discriminator_optimizer.step();
      // train generator
      // discriminator should assign probabilities close to 1
      ;
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      auto g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();
      {

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("") << (" ") << (std::setw(8)) << (" epoch='")
                    << (epoch) << ("'") << (std::setw(8))
                    << (" (batch_index)++='") << ((batch_index)++) << ("'")
                    << (std::setw(8)) << (" real_d_loss.item<float>()='")
                    << (real_d_loss.item<float>()) << ("'") << (std::setw(8))
                    << (" fake_d_loss.item<float>()='")
                    << (fake_d_loss.item<float>()) << ("'") << (std::setw(8))
                    << (" d_loss.item<float>()='") << (d_loss.item<float>())
                    << ("'") << (std::setw(8)) << (" g_loss.item<float>()='")
                    << (g_loss.item<float>()) << ("'") << (std::endl)
                    << (std::flush);
      }
      if ((0) == (batch_index % kCheckpointEvery)) {
        torch::save(generator, "generator.pt");
        torch::save(generator_optimizer, "generator_optimizer.pt");
        torch::save(discriminator, "discriminator.pt");
        torch::save(discriminator_optimizer, "discriminator_optimizer.pt");
        auto samples =
            generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
        torch::save((((0.50f)) * (((samples) + ((1.0f))))),
                    torch::str("dcgan-sample-", (checkpoint_counter)++, ".pt"));
        {

          (std::cout) << (std::setw(10))
                      << (std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count())
                      << (" ") << (std::this_thread::get_id()) << (" ")
                      << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ") << ("") << (" ") << (std::setw(8))
                      << (" checkpoint_counter='") << (checkpoint_counter)
                      << ("'") << (std::endl) << (std::flush);
        }
      }
    }
  }
  return 0;
}