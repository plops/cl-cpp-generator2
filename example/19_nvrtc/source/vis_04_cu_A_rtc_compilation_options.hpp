#ifndef VIS_04_CU_A_RTC_COMPILATION_OPTIONS_H
#define VIS_04_CU_A_RTC_COMPILATION_OPTIONS_H
#include "utils.h"
;
#include "globals.h"
;
#include <algorithm>
#include <vector>
;
#include "vis_02_cu_A_device.hpp"
;
#include "vis_04_cu_A_rtc_compilation_options.hpp"
;
class CompilationOptions  {
            std::vector<std::string> _options ;
    mutable std::vector<const char*> _chOptions ;
        public:
        void insert (const std::string& op)  ;  ;
        void insert (const std::string& name, const std::string& value)  ;  ;
        template<typename T> void insertOptions (const T& p)  ;  ;
        template<typename T, typename... TS> void insertOptions (const T& p, const TS& ...ts)  ;  ;
        template<typename... TS> CompilationOptions (TS&& ...ts)  ;  ;
            CompilationOptions()=default;
        size_t numOptions () const ;  ;
        const char** options () const ;  ;
};
class GpuArchitecture  {
            const std::string _arch ;
        public:
         GpuArchitecture (int major, int minor)  ;  ;
         GpuArchitecture (const CudaDeviceProperties& props)  ;  ;
        std::string name () const ;  ;
        const std::string& value () const ;  ;
};
class CPPLang  {
            CPPLangVer _version ;
        public:
         CPPLang (CPPLangVer version)  ;  ;
        auto name () const ;  ;
        auto value () const ;  ;
};
#endif