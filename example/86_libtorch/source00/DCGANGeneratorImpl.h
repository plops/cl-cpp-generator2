#ifndef DCGANGENERATORIMPL_H
#define DCGANGENERATORIMPL_H

#include <torch/autograd.h>
#include <torch/cuda.h>
#include <torch/nn.h>
#include <torch/types.h>
class DCGANGeneratorImpl : public torch::nn::Module {
        public:
        torch::nn::ConvTranspose2d conv1;
        torch::nn::ConvTranspose2d conv2;
        torch::nn::ConvTranspose2d conv3;
        torch::nn::ConvTranspose2d conv4;
        torch::nn::BatchNorm2d batch_norm1;
        torch::nn::BatchNorm2d batch_norm2;
        torch::nn::BatchNorm2d batch_norm3;
         DCGANGeneratorImpl (int kNoiseSize)     ;  
        torch::Tensor forward (torch::Tensor x)     ;  
};

#endif /* !DCGANGENERATORIMPL_H */