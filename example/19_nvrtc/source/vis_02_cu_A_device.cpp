
#include "utils.h"

#include "globals.h"

;
extern State state;
#include "vis_02_cu_A_device.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
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
}
CudaDeviceProperties CudaDeviceProperties::ByIntegratedType(bool integrated) {
  auto props = cudaDeviceProp{0};
  props.integrated = (integrated) ? (1) : (0);
  return FromExistingProperties(props);
}
const auto &CudaDeviceProperties::getRawStruct() const { return _props; }
int CudaDeviceProperties::major() const { return _props.major; }
int CudaDeviceProperties::minor() const { return _props.minor; }
bool CudaDeviceProperties::integrated() const {
  return (0) < (_props.integrated);
}
const char *CudaDeviceProperties::name() const { return _props.name; };
CudaDevice::CudaDevice(int device) : _device(device), _props(device) {}
CUdevice CudaDevice::handle() const {
  CUdevice h;
  if (!((CUDA_SUCCESS) == (cuDeviceGet(&h, _device)))) {
    throw std::runtime_error("cuDeviceGet(&h, _device)");
  };
  return h;
}
CudaDevice CudaDevice::FindByProperties(const CudaDeviceProperties &props) {
  int device;
  if (!((cudaSuccess) == (cudaChooseDevice(&device, &props.getRawStruct())))) {
    throw std::runtime_error(
        "cudaChooseDevice(&device, &props.getRawStruct())");
  };
  return CudaDevice{device};
}
int CudaDevice::NumberOfDevices() {
  int numDevices = 0;
  if (!((cudaSuccess) == (cudaGetDeviceCount(&numDevices)))) {
    throw std::runtime_error("cudaGetDeviceCount(&numDevices)");
  };
  return numDevices;
}
void CudaDevice::setAsCurrent() { cudaSetDevice(_device); }
const auto &CudaDevice::properties() const { return _props; }
const char *CudaDevice::name() const { return properties().name(); }
CudaDevice CudaDevice::FindByName(std::string name) {
  auto numDevices = NumberOfDevices();
  if ((numDevices) == (0)) {
    throw std::runtime_error("no cuda devices found");
  };
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  for (int i = 0; (i) < (numDevices); (i) += (1)) {
    auto devi = CudaDevice(i);
    auto deviName = std::string(devi.name());
    std::transform(deviName.begin(), deviName.end(), deviName.begin(),
                   ::tolower);
    if (!((std::string::npos) == (deviName.find(name)))) {
      return devi;
    };
  }
  throw std::runtime_error("could not find cuda device by name");
}
std::vector<CudaDevice> CudaDevice::EnumerateDevices() {
  std::vector<CudaDevice> res;
  auto n = NumberOfDevices();
  for (int i = 0; (i) < (n); (i) += (1)) {
    res.emplace_back(i);
  }
  return res;
}
CudaDevice CudaDevice::CurrentDevice() {
  int device;
  if (!((cudaSuccess) == (cudaGetDevice(&device)))) {
    throw std::runtime_error("cudaGetDevice(&device)");
  };
  return CudaDevice{device};
};