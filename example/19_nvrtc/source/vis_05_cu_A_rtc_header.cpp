
#include "utils.h"

#include "globals.h"

;
extern State state;
#include "vis_01_cu_A_rtc_code.hpp"

#include "vis_05_cu_A_rtc_header.hpp"
template <typename... ARGS>
Header::Header(const std::string &name, ARGS &&... args)
    : Code(std::forward<ARGS>(args)...), _name(name) {}
const std::string &Header::name() const { return _name; };