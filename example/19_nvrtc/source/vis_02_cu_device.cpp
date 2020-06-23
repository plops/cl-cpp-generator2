
#include "utils.h"

#include "globals.h"

;
extern State state;
//  g++ --std=gnu++20 vis_02_cu_device.cpp -I /media/sdb4/cuda/11.0.1/include/
#include "vis_02_cu_device.hpp"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
CudaDeviceProperties::CudaDeviceProperties(const cudaDeviceProp &props)
    : _props(props) {}
CudaDeviceProperties::CudaDeviceProperties(int device) {
  cudaGetDeviceProperties(&_props, device);
  auto nameSize = ((sizeof(_props.name)) / (sizeof(_props.name[0])));
  _props.name[((nameSize) - (1))] = '\0';
}
CudaDeviceProperties
CudaDeviceProperties::FromExistingProperties(const cudaDeviceProp &props) {
  return CudaDeviceProperties{props};
};
CudaDevice::CudaDevice(int device) : _device(device), _props(device) {}
CUdevice CudaDevice::handle() const {
  CUdevice h;
  if (!((CUDA_SUCCESS) == (cuDeviceGet(&h, _device)))) {
    throw std::runtime_error("cuDeviceGet(&h, _device)");
  };
  return h;
};