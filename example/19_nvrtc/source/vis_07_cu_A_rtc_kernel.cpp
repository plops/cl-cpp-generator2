
#include "utils.h"

#include "globals.h"

;
extern State state;

#include <nvrtc.h>

#include "vis_08_cu_A_rtc_module.hpp"

#include "vis_07_cu_A_rtc_kernel.hpp"

;
//template <typename T>;
//template <typename T, T y>;
void TemplateParameters::addComma() {
  if (_first) {
    _first = false;
  } else {
    _val = ((_val) + (","));
  }
}
template <typename T> auto &TemplateParameters::addValue(const T &val) {
  addComma();
  _val = ((_val) + (std::string(val)));
  return *this;
}
template <typename T> auto &TemplateParameters::addType() {
  addComma();
  _val = ((_val) + (NameExtractor<T>::extract()));
  return *this;
}
const std::string &TemplateParameters::operator()() const { return _val; };
Kernel::Kernel(const std::string &name) : _name(name) {}
Kernel &Kernel::instantiate(const TemplateParameters &tp) {
  _name = ((_name) + ("<") + (tp()) + (">"));
  return *this;
}
template <typename... ARGS> Kernel &Kernel::instantiate() {
  TemplateParameters tp;
  AddTypesToTemplate<ARGS...>(tp);
  return instantiate(tp);
}
const std::string &Kernel::name() const { return _name; };
;