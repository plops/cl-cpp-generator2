
#include "utils.h"

#include "globals.h"

;
extern State state;

#include "vis_08_cu_A_rtc_module.hpp"

#include "vis_07_cu_A_rtc_kernel.hpp"

inline Kernel::Kernel(const std::string &name) : _name(name) {}
inline Kernel &Kernel::instantiate(const TemplateParameters &tp) {
  _name = ((_name) + ("<") + (tp()) + (">"));
  return *this;
}
const std::string &Kernel::name() const { return _name; };
static inline void AddTypesToTemplate(TemplateParameters &params) {}
template <typename T>
static inline void AddTypesToTemplate(TemplateParameters &params) {
  params.addType<T>();
}
template <typename T, typename U, typename... REST>
static inline void AddTypesToTemplate(TemplateParameters &params) {
  params.addType<T>();
  AddTypesToTemplate<U, REST...>(params);
};