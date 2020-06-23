
#include "utils.h"

#include "globals.h"

;
extern State state;
#include <cuda.h>
#include <fstream>
#include <nvrtc.h>
#include <streambuf>
#include <string>

class Module;
class Program;

#include "vis_01_rtc.hpp"

// Code c{ <params> };  .. initialize
// Code c = Code::FromFile(fname);  .. load contents of file
// auto& code = c.code() .. get reference to internal string
;
template <typename... ARGS>
Code::Code(ARGS &&... args) : _code(std::forward<ARGS>(args)...) {}
Code Code::FromFile(const std::string &name) {
  auto input = std::ifstream(name);
  if (!(input.good())) {
    throw std::runtime_error("can't read file");
  };
  input.seekg(0, std::ios::end);
  std::string str;
  str.reserve(input.tellg());
  input.seekg(0, std::ios::beg);
  str.assign(std::istreambuf_iterator<char>(input),
             std::istreambuf_iterator<char>());
  return Code{std::move(str)};
}
const auto &Code::code() const { return _code; };
template <typename... ARGS>
Header::Header(const std::string &name, ARGS &&... args)
    : Code(std::forward<ARGS>(args)...), _name(name) {}
const auto &Header::name() const { return _name; };
template <typename... ARGS>
std::vector<void *> BuildArgs(const ARGS &... args) {
  return {const_cast<void *>(reinterpret_cast<const void *>(&args))...};
}
template <typename T>
std::string NameExtractor<T>::extract() {
  std::string type_name;
  nvrtcGetTypeName<T>(&type_name);
  return type_name;
};
template <typename T, T y>
static std::string
NameExtractor<std::integral_constant<T, y>>::extract() {
  return std::to_string(y);
};
inline Kernel::Kernel(const std::string &name) : _name(name) {}
inline Kernel &Kernel::instantiate(const TemplateParameters &tp) {
  _name = ((_name) + ("<") + (tp()) + (">"));
  return *this;
}
template <typename... ARGS> Kernel &Kernel::instantiate() {
  TemplateParameters tp;
  AddTypesToTemplate<ARGS...>(tp);
  return instantiate(tp);
}
const auto &Kernel::name() const { return _name; }
void Kernel::init(const Module &m, const Program &p) {
  if (!((CUDA_SUCCESS) ==
        (cuModuleGetFunction(&_kernel, m.module(),
                             p.loweredName(*this).c_str())))) {
    throw std::runtime_error("cuModuleGetFunction(&_kernel, m.module(), "
                             "p.loweredName(*this).c_str())");
  };
};
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