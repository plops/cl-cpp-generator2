
#include "utils.h"

#include "globals.h"

;
extern State state;
//  g++ --std=gnu++20 vis_02_cu_device.cpp -I /media/sdb4/cuda/11.0.1/include/
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
class CudaDeviceProperties {
  cudaDeviceProp _props;
  explicit CudaDeviceProperties(const cudaDeviceProp &props) : _props(props) {}

public:
  CudaDeviceProperties(int device) {
    cudaGetDeviceProperties(&_props, device);
    auto nameSize = ((sizeof(_props.name)) / (sizeof(_props.name[0])));
    _props.name[((nameSize) - (1))] = '\0';
  }
  static CudaDeviceProperties
  FromExistingProperties(const cudaDeviceProp &props) {
    return CudaDeviceProperties{props};
  }
  static CudaDeviceProperties ByIntegratedType(bool integrated) {
    auto props = cudaDeviceProp{0};
    props.integrated = (integrated) ? (1) : (0);
    return FromExistingProperties(props);
  }
  const auto &getRawStruct() const { return _props; }
  auto major() const { return _props.major; }
  auto minor() const { return _props.minor; }
  bool integrated() const { return (0) < (_props.integrated); }
  const char *name() const { return _props.name; }
};
class CudaDevice {
  int _device;
  CudaDeviceProperties _props;

public:
  explicit CudaDevice(int device) : _device(device), _props(device) {}
  inline CUdevice handle() const {
    CUdevice h;
    if (!((CUDA_SUCCESS) == (cuDeviceGet(&h, _device)))) {
      throw std::runtime_error("cuDeviceGet(&h, _device)");
    };
    return h;
  }
  static CudaDevice FindByProperties(const CudaDeviceProperties &props) {
    int device;
    if (!((cudaSuccess) ==
          (cudaChooseDevice(&device, &props.getRawStruct())))) {
      throw std::runtime_error(
          "cudaChooseDevice(&device, &props.getRawStruct())");
    };
    return CudaDevice{device};
  }
  static int NumberOfDevices() {
    int numDevices = 0;
    if (!((cudaSuccess) == (cudaGetDeviceCount(&numDevices)))) {
      throw std::runtime_error("cudaGetDeviceCount(&numDevices)");
    };
    return numDevices;
  }
  void setAsCurrent() { cudaSetDevice(_device); }
  const auto &properties() const { return _props; }
  const char *name() const { return properties().name(); }
  static CudaDevice FindByName(std::string name) {
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
  static std::vector<CudaDevice> EnumerateDevices() {
    std::vector<CudaDevice> res;
    auto n = NumberOfDevices();
    for (int i = 0; (i) < (n); (i) += (1)) {
      res.emplace_back(i);
    }
    return res;
  }
  static CudaDevice CurrentDevice() {
    int device;
    if (!((cudaSuccess) == (cudaGetDevice(&device)))) {
      throw std::runtime_error("cudaGetDevice(&device)");
    };
    return CudaDevice{device};
  }
};
class CudaContext {
  CUcontext _ctx;

public:
  CudaContext(const CudaDevice &device) : _ctx(nullptr) {
    if (!((CUDA_SUCCESS) == (cuInit(0)))) {
      throw std::runtime_error("cuInit(0)");
    };
    if (!((CUDA_SUCCESS) == (cuCtxCreate(&_ctx, 0, device.handle())))) {
      throw std::runtime_error("cuCtxCreate(&_ctx, 0, device.handle())");
    };
  }
  ~CudaContext() {
    if (_ctx) {
      cuCtxDestroy(_ctx);
    };
  }
};