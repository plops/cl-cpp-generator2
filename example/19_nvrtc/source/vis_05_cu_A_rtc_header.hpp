#ifndef VIS_05_CU_A_RTC_HEADER_H
#define VIS_05_CU_A_RTC_HEADER_H
#include "utils.h"
;
#include "globals.h"
;
#include "vis_01_cu_A_rtc_code.hpp"
;
#include "vis_05_cu_A_rtc_header.hpp"
;
class Header : public Code {
            const std::string _name ;
        public:
        template<typename... ARGS>  Header (const std::string& name, ARGS&& ...args)  ;  ;
        const auto& name () const ;  ;
};
#endif