#include "DCGANGeneratorImpl.h"
#include <iostream>
#include <popl.hpp>
#include <spdlog/spdlog.h>
#include <torch/data.h>
#include <torch/optim.h>
TORCH_MODULE(DCGANGenerator);
int main(int argc, char **argv) {
  spdlog::info("start  argc='{}'", argc);
  auto op = popl::OptionParser("allowed opitons");
  auto kNoiseSize = int(12);
  auto kBatchSize = int(32);
  auto kNumberOfEpochs = int(10);
  auto helpOption = op.add<popl::Switch>("h", "help", "produce help message");
  auto verboseOption =
      op.add<popl::Switch>("v", "verbose", "produce verbose output");
  auto kNoiseSizeOption =
      op.add<popl::Value<int>>("n", "kNoiseSize", "parameter", 12, &kNoiseSize);
  auto kBatchSizeOption =
      op.add<popl::Value<int>>("b", "kBatchSize", "parameter", 32, &kBatchSize);
  auto kNumberOfEpochsOption = op.add<popl::Value<int>>(
      "e", "kNumberOfEpochs", "parameter", 10, &kNumberOfEpochs);
  op.parse(argc, argv);
  if (helpOption->count()) {
    (std::cout) << (op) << (std::endl);
    exit(0);
  }
  torch::Tensor tensor = torch::eye(3);
  (std::cout) << (tensor) << (std::endl);
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
  auto dataset =
      torch::data::datasets::MNIST("./mnist")
          .map(torch::data::transforms::Normalize<>((0.50f), (0.50f)))
          .map(torch::data::transforms::Stack<>());
  auto batches_per_epoch =
      ((std::ceil(dataset.size().value())) / (static_cast<double>(kBatchSize)));
  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));
  for (auto &batch : *data_loader) {
    spdlog::info("  batch.data.size(0)='{}'", batch.data.size(0));
    for (auto i = 0; (i) < (batch.data.size(0)); (i) += (1)) {
      spdlog::info("  batch.target[i].item<int64_t>()='{}'",
                   batch.target[i].item<int64_t>());
    }
  }
  auto generator_optimizer = torch::optim::Adam(
      generator->parameters(),
      torch::optim::AdamOptions((2.00e-4f)).betas({(0.90f), (0.50f)}));
  auto discriminator_optimizer = torch::optim::Adam(
      generator->parameters(),
      torch::optim::AdamOptions((5.00e-4f)).betas({(0.90f), (0.50f)}));
  for (auto epoch = 0; (epoch) < (kNumberOfEpochs); (epoch) += (1)) {
    auto batch_index = int64_t(0);
    for (auto &batch : *data_loader) {
      auto noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1});
      // train discriminator with real images
      auto real_images = batch.data;
      auto real_labels =
          torch::empty(batch.data.size(0)).uniform_((0.80f), (1.0f));
      auto real_output = discriminator->forward(real_images);
      auto real_d_loss = torch::binary_cross_entropy(real_output, real_labels);
      real_d_loss.backward();
      // train discriminator with fake images
      auto fake_images = generator->forward(noise);
      auto fake_labels =
          torch::empty(batch.data.size(0)).uniform_((0.80f), (1.0f));
      auto fake_output = discriminator->forward(fake_images);
      auto fake_d_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      fake_d_loss.backward();
      auto d_loss = ((fake_d_loss) + (real_d_loss));
      discriminator_optimizer.step();
      {
        // train generator
        generator->zero_grad();
        fake_labels.fill_(1);
        fake_output = discriminator->forward(fake_images);
        auto g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
        g_loss.backward();
        generator_optimizer.step();
        spdlog::info("  epoch='{}'  kNumberOfEpochs='{}'  batch_index='{}'  "
                     "batches_per_epoch='{}'  d_loss.item<float>()='{}'  "
                     "g_loss.item<float>()='{}'",
                     epoch, kNumberOfEpochs, batch_index, batches_per_epoch,
                     d_loss.item<float>(), g_loss.item<float>());
        (batch_index)++;
      }
    }
  }
}