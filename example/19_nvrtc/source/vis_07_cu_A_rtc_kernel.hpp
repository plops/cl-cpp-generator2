#ifndef VIS_07_CU_A_RTC_KERNEL_H
#define VIS_07_CU_A_RTC_KERNEL_H
#include "utils.h"
;
#include "globals.h"
;
#include "vis_08_cu_A_rtc_module.hpp"
;
#include "vis_07_cu_A_rtc_kernel.hpp"
;
class Kernel  {
            CUfunction _kernel  = nullptr;
    std::string _name ;
        public:
        inline Kernel (const std::string& name)  ;  ;
        inline Kernel& instantiate (const TemplateParameters& tp)  ;  ;
        template<template<typename... ARGS>> inline Kernel& instantiate ()    {
                        TemplateParameters tp ;
        AddTypesToTemplate<ARGS...>(tp);
        return instantiate(tp);
};
        const std::string& name () const ;  ;
};
class TemplateParameters  {
            std::string _val ;
    auto _first  = true;
    void addComma ()  ;  ;
    public:
    template<typename T> auto& addValue (const T& val)  ;  ;
    template<typename T> auto& addType ()  ;  ;
    const std::string& operator() () const ;  ;
};
static inline void AddTypesToTemplate (TemplateParameters& params)  ;  
template<typename T> static inline void AddTypesToTemplate (TemplateParameters& params)  ;  
template<typename T, typename U, typename... REST> static inline void AddTypesToTemplate (TemplateParameters& params)  ;  
#endif