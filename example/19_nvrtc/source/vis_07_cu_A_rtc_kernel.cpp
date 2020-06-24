
#include "utils.h"

#include "globals.h"

;
extern State state;
#include "vis_07_cu_A_rtc_kernel.hpp"
#include "vis_08_cu_A_rtc_module.hpp"
inline Kernel::Kernel(const std::string &name) : _name(name) {}
const std::string &Kernel::name() const { return _name; };