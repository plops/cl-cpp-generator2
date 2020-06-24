#ifndef VIS_03_CU_A_CONTEXT_H
#define VIS_03_CU_A_CONTEXT_H
#include "utils.h"
;
#include "globals.h"
;
#include <cuda.h>
;
#include "vis_02_cu_A_device.hpp"
;
#include "vis_03_cu_A_context.hpp"
;
class CudaContext  {
            CUcontext _ctx ;
        public:
         CudaContext (const CudaDevice& device)  ;  ;
         ~CudaContext ()  ;  ;
};
#endif