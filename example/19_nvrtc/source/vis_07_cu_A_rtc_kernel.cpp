
#include "utils.h"

#include "globals.h"

;
extern State state;

#include <nvrtc.h>

#include "vis_08_cu_A_rtc_module.hpp"

#include "vis_07_cu_A_rtc_kernel.hpp"

template <typename... ARGS>
std::vector<void *> BuildArgs(const ARGS &... args) {
  return {const_cast<void *>(reinterpret_cast<const void *>(&args))...};
}
template <typename T> std::string NameExtractor<T>::extract() {
  std::string type_name;
  nvrtcGetTypeName<T>(&type_name);
  return type_name;
};
template <typename T, T y>
std::string NameExtractor<std::integral_constant<T, y>>::extract() {
  return std::to_string(y);
};
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