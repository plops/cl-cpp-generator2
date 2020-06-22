
#include "utils.h"

#include "globals.h"

;
extern State state;
#include "vis_02_cu_device.hpp"
#include "vis_03_cu_program.hpp"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
Module::Module(const CudaContext &ctx, const Program &p) {
  cuModuleLoadDataEx(&_module, p.PTX().c_str(), 0, 0, 0);
}
auto Module::module() const { return _module; };