#ifndef VIS_01_RTC_H
#define VIS_01_RTC_H
#include "utils.h"
;
#include "globals.h"
;
#include <nvrtc.h>
#include <cuda.h>
#include <string>
#include <fstream>
#include <streambuf>
;
#include "vis_04_cu_module.hpp"
;
#include "vis_02_cu_device.hpp"
;
#include "vis_01_rtc.hpp"
;
class Code  {
            const std::string _code ;
        public:
        template<typename... ARGS> Code (ARGS&& ...args);  
        static Code FromFile (const std::string& name);  
        const auto& code ();  
};
class Header : public Code {
            const std::string _name ;
        public:
        template<typename... ARGS> Header (const std::string& name, ARGS&& ...args);  
        const auto& name ();  
};
template<typename... ARGS> static inline std::vector<void*> BuildArgs (const ARGS& ...args);  
template<typename T> class NameExtractor  {
                                public:
                                static std::string extract ();  
};
template<typename T, T y> class NameExtractor<std::integral_constant<T, y>>  {
                                public:
                                static std::string extract ();  
};
class TemplateParameters  {
                        std::string _val ;
        bool _first  = true;
        void addComma ();  
        public:
        template<typename T> auto& addValue (const T& val);  
        template<typename T> auto& addType ();  
        const std::string& operator() ();  ;
};
class Kernel  {
            CUfunction _kernel  = nullptr;
    std::string _name ;
        public:
        inline Kernel (const std::string& name);  
         ;
        inline Kernel& instantiate (const TemplateParameters& tp);  
        template<typename... ARGS> Kernel& instantiate ();  ;
        const auto& name ();  
        void init (const Module& m, const Program& p);  
};
class CompilationOptions  {
            std::vector<std::string> _options ;
    mutable std::vector<const char*> _chOptions ;
        public:
        void insert (const std::string& op);  
        void insert (const std::string& name, const std::string& value);  
        template<typename T> void insertOptions (const T& p);  
        template<typename T, typename... TS> void insertOptions (const T& p, const TS& ...ts);  
        template<typename... TS> CompilationOptions (TS&& ...ts);  
            CompilationOptions()=default;
        auto numOptions ();  
        const char** options ();  
};
class GpuArchitecture  {
                                    const std::string _arch ;
                        public:
                         GpuArchitecture (int major, int minor);  
                         GpuArchitecture (const CudaDeviceProperties& props);  
                        auto name ();  
                        auto& value ();  
};
class CPPLang  {
                                    CPPLangVer _version ;
                        public:
                         CPPLang (CPPLangVer version);  
                        auto name ();  
                        auto value ();  
};
static inline void AddTypesToTemplate (Kernel::TemplateParameters& params);  
template<typename T> static inline void AddTypesToTemplate (Kernel::TemplateParameters& params);  
template<typename T, typename U, typename... REST> static inline void AddTypesToTemplate (Kernel::TemplateParameters& params);  
template<typename... ARGS> inline Kernel& Kernel::instantiate ();  
#endif