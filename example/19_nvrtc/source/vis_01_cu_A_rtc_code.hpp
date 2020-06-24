#ifndef VIS_01_CU_A_RTC_CODE_H
#define VIS_01_CU_A_RTC_CODE_H
#include "utils.h"
;
#include "globals.h"
;
#include <string>
#include <fstream>
#include <streambuf>
;
#include "vis_01_cu_A_rtc_code.hpp"
;
class Code  {
            const std::string _code ;
        public:
        template<typename... ARGS> explicit  Code (ARGS&& ...args)  ;  ;
        static Code FromFile (const std::string& name)  ;  ;
        const auto& code () const ;  ;
};
#endif