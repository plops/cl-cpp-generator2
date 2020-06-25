
#include "utils.h"

#include "globals.h"

;
extern State state;

#include <algorithm>
#include <vector>

#include "vis_02_cu_A_device.hpp"

#include "vis_04_cu_A_rtc_compilation_options.hpp"
void CompilationOptions::insert(const std::string &op) {
  _options.push_back(op);
}
void CompilationOptions::insert(const std::string &name,
                                const std::string &value) {
  if (value.empty()) {
    insert(name);
  } else {
    _options.push_back(((name) + ("=") + (value)));
  }
}
template <typename T> void CompilationOptions::insertOptions(const T &p) {
  insert(p.name(), p.value());
}
template <typename T, typename... TS>
void CompilationOptions::insertOptions(const T &p, const TS &... ts) {
  insert(p.name(), p.value());
  insertOptions(ts...);
}
template <typename... TS> CompilationOptions::CompilationOptions(TS &&... ts) {
  insertOptions(ts...);
}
size_t CompilationOptions::numOptions() const { return _options.size(); }
const char **CompilationOptions::options() const {
  _chOptions.resize(_options.size());
  std::transform(_options.begin(), _options.end(), _chOptions.begin(),
                 [](const auto &s) { return s.c_str(); });
  return _chOptions.data();
};
GpuArchitecture::GpuArchitecture(int major, int minor)
    : _arch(((std::string("compute_")) + (std::to_string(major)) +
             (std::to_string(minor)))) {}
GpuArchitecture::GpuArchitecture(const CudaDeviceProperties &props)
    : GpuArchitecture(props.major(), props.minor()) {}
std::string GpuArchitecture::name() const { return "--gpu-architecture"; }
const std::string &GpuArchitecture::value() const { return _arch; };

CPPLang::CPPLang(CPPLangVer version) : _version(version) {}
auto CPPLang::name() const { return "--std"; }
auto CPPLang::value() const {
  switch (_version) {
  case CPP_x11: {
    return "c++11";
  }
  case CPP_x14: {
    return "c++14";
  }
  case CPP_x17: {
    return "c++17";
  }
  }
  throw std::runtime_error("unknown C++ version");
};