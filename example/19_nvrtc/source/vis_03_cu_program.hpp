#ifndef VIS_03_CU_PROGRAM_H
#define VIS_03_CU_PROGRAM_H
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
#include "vis_01_rtc.hpp"
;
class Program  {
            nvrtcProgram _prog ;
        public:
         Program (const std::string& name, const Code& code, const std::vector<Header>& headers);  
         Program (const std::string& name, const Code& code);  
        inline void registerKernel (const Kernel& k);  
        void compile (const CompilationOptions& opt = {});  
        inline std::string PTX ();  
};
#endif