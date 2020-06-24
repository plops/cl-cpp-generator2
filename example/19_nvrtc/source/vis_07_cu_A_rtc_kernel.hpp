#ifndef VIS_07_CU_A_RTC_KERNEL_H
#define VIS_07_CU_A_RTC_KERNEL_H
#include "utils.h"
;
#include "globals.h"
;
#include "vis_07_cu_A_rtc_kernel.hpp"
;
class TemplateParameters  {
                        std::string _val ;
        bool _first  = true;
        void addComma ()  ;  ;
        public:
        const std::string& operator() () const ;  ;
};
class Kernel  {
            CUfunction _kernel  = nullptr;
    std::string _name ;
        public:
        inline Kernel (const std::string& name)  ;  ;
         ;
        const auto& name () const ;  ;
};
#endif