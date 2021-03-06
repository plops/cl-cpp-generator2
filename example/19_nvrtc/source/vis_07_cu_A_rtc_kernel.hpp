#ifndef VIS_07_CU_A_RTC_KERNEL_H
#define VIS_07_CU_A_RTC_KERNEL_H
#include "utils.h"
;
#include "globals.h"
;
#include <nvrtc.h>
;
#include "vis_08_cu_A_rtc_module.hpp"
;
#include "vis_07_cu_A_rtc_kernel.hpp"
;
template<typename... ARGS> static inline std::vector<void*> BuildArgs (const ARGS& ...args)    {
        return {const_cast<void*>(reinterpret_cast<const void*>(&args)) ...};
};;
class TemplateParameters  {
            std::string _val ;
    bool _first  = true;
        public:
        void addComma ()  ;  ;
        template<typename T> auto& addValue (const T& val)  ;  ;
        template<typename T> auto& addType ()  ;  ;
        const std::string& operator() () const ;  ;
};
class Kernel  {
            CUfunction _kernel  = nullptr;
    std::string _name ;
        public:
         Kernel (const std::string& name)  ;  ;
        Kernel& instantiate (const TemplateParameters& tp)  ;  ;
        template<typename... ARGS> Kernel& instantiate ()  ;  ;
        const std::string& name () const ;  ;
};
static inline void AddTypesToTemplate (TemplateParameters& params)    {
};
template<typename T> static inline void AddTypesToTemplate (TemplateParameters& params)    {
        params.addType<T>();
};
template<typename T, typename U, typename... REST> static inline void AddTypesToTemplate (TemplateParameters& params)    {
        params.addType<T>();
        AddTypesToTemplate<U, REST...>(params);
};;
#endif