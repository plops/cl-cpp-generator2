#ifndef VIS_08_CU_A_RTC_MODULE_H
#define VIS_08_CU_A_RTC_MODULE_H
#include "utils.h"
;
#include "globals.h"
;
#include <cuda.h>
;
#include "vis_06_cu_A_rtc_program.hpp"
;
#include "vis_03_cu_A_context.hpp"
;
#include "vis_08_cu_A_rtc_module.hpp"
;
class Module  {
            CUmodule _module ;
        public:
         Module (const CudaContext& ctx, const Program& p)  ;  ;
        CUmodule module () const ;  ;
};
#endif