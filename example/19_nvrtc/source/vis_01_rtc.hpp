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
#include "vis_01_rtc.hpp"
;
class Code  {
            const std::string _code ;
        public:
        template<typename... ARGS> Code (ARGS&& ...args)  ;  
        static Code FromFile (const std::string& name)  ;  
        const auto& code () const ;  
};
class Header : public Code {
            const std::string _name ;
        public:
        template<typename... ARGS> Header (const std::string& name, ARGS&& ...args)  ;  
        const auto& name () const ;  
};
template<typename... ARGS> static inline std::vector<void*> BuildArgs (const ARGS& ...args)  ;  
template<typename T> class NameExtractor  {
        public:
        static std::string extract ()  ;  
};
template<typename T, T y> class NameExtractor<std::integral_constant<T, y>>  {
        public:
        static std::string extract ()  ;  
};
class TemplateParameters  {
                        std::string _val ;
        bool _first  = true;
        void addComma ()  ;  
        public:
        template<typename T> auto& addValue (const T& val)  ;  
        template<typename T> auto& addType ()  ;  
        const std::string& operator() () const ;  ;
};
class Kernel  {
            CUfunction _kernel  = nullptr;
    std::string _name ;
        public:
        inline Kernel (const std::string& name)  ;  
         ;
        inline Kernel& instantiate (const TemplateParameters& tp)  ;  
        template<typename... ARGS> inline Kernel& instantiate ()  ;  
        const auto& name () const ;  
        void init (const Module& m, const Program& p)  ;  
};
static inline void AddTypesToTemplate (TemplateParameters& params)  ;  
template<typename T> static inline void AddTypesToTemplate (TemplateParameters& params)  ;  
template<typename T, typename U, typename... REST> static inline void AddTypesToTemplate (TemplateParameters& params)  ;  
#endif