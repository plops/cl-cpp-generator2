
#include "utils.h"

#include "globals.h"

;
extern State state;
#include "vis_07_cu_A_rtc_kernel.hpp"
inline Kernel::Kernel(const std::string &name) : _name(name) {}
const std::string &Kernel::name() const { return _name; };