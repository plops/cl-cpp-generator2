#ifndef VIS_06_CU_A_RTC_PROGRAM_H
#define VIS_06_CU_A_RTC_PROGRAM_H
#include "utils.h"
;
#include "globals.h"
;
#include <nvrtc.h>
;
#include "vis_01_cu_A_rtc_code.hpp"
;
#include "vis_04_cu_A_rtc_compilation_options.hpp"
;
#include "vis_05_cu_A_rtc_header.hpp"
;
#include "vis_06_cu_A_rtc_program.hpp"
;
class Kernel;;
class Program  {
            nvrtcProgram _prog ;
        public:
         Program (const std::string& name, const Code& code, const std::vector<Header>& headers)  ;  ;
         Program (const std::string& name, const Code& code)  ;  ;
        void compile (const CompilationOptions& opt = {})  ;  ;
        inline std::string PTX () const   {
                        std::size_t size  = 0;
        if ( !((NVRTC_SUCCESS)==(nvrtcGetPTXSize(_prog, &size))) ) {
                                    throw std::runtime_error("nvrtcGetPTXSize(_prog, &size)");
};
                auto str  = std::string(size, '\0');
        if ( !((NVRTC_SUCCESS)==(nvrtcGetPTX(_prog, &str.front()))) ) {
                                    throw std::runtime_error("nvrtcGetPTX(_prog, &str.front())");
};
        return str;
};
        void registerKernel (const Kernel& k)  ;  ;
};
#endif