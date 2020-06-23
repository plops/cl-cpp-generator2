#ifndef VIS_04_CU_MODULE_H
#define VIS_04_CU_MODULE_H
#include "utils.h"
;
#include "globals.h"
;
#include <cuda_runtime.h>
#include <cuda.h>
;
#include <algorithm>
#include <vector>
;
#include "vis_03_cu_program.hpp"
;
#include "vis_04_cu_module.hpp"
;
class Module  {
            CUmodule _module ;
        public:
         Module (const CudaContext& ctx, const Program& p)  ;  
        CUmodule module () const ;  
};
#endif