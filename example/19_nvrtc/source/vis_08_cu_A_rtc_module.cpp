
#include "utils.h"

#include "globals.h"

;
extern State state;
#include <cuda.h>

#include "vis_03_cu_A_context.hpp"
#include "vis_06_cu_A_rtc_program.hpp"

#include "vis_08_cu_A_rtc_module.hpp"

Module::Module(const CudaContext &ctx, const Program &p) {
  cuModuleLoadDataEx(&_module, p.PTX().c_str(), 0, 0, 0);
}
CUmodule Module::module() const { return _module; };