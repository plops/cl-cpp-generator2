
#include "utils.h"

#include "globals.h"

;
extern State state;

#include <cuda.h>

#include "vis_02_cu_A_device.hpp"

#include "vis_03_cu_A_context.hpp"
CudaContext::CudaContext(const CudaDevice &device) : _ctx(nullptr) {
  if (!((CUDA_SUCCESS) == (cuInit(0)))) {
    throw std::runtime_error("cuInit(0)");
  };
  if (!((CUDA_SUCCESS) == (cuCtxCreate(&_ctx, 0, device.handle())))) {
    throw std::runtime_error("cuCtxCreate(&_ctx, 0, device.handle())");
  };
}
CudaContext::~CudaContext() {
  if (_ctx) {
    cuCtxDestroy(_ctx);
  };
};