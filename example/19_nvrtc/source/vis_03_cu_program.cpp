
#include "utils.h"

#include "globals.h"

;
extern State state;
#include "vis_01_rtc.hpp"
#include "vis_03_cu_program.hpp"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
Program::Program(const std::string &name, const Code &code,
                 const std::vector<Header> &headers) {
  auto nh = headers.size();
  std::vector<const char *> headersContent;
  std::vector<const char *> headersNames;
  for (auto &h : headers) {
    headersContent.push_back(h.code().c_str());
    headersContent.push_back(h.name().c_str());
  };
  if (!((NVRTC_SUCCESS) ==
        (nvrtcCreateProgram(
            &_prog, code.code().c_str(), name.c_str(), static_cast<int>(nh),
            ((0) < (nh)) ? (headersContent.data()) : (nullptr),
            ((0) < (nh)) ? (headersNames.data()) : (nullptr))))) {
    throw std::runtime_error(
        "nvrtcCreateProgram(&_prog, code.code().c_str(), name.c_str(), "
        "static_cast<int>(nh), ((0)<(nh)) ? (headersContent.data()) : "
        "(nullptr), ((0)<(nh)) ? (headersNames.data()) : (nullptr))");
  };
}
Program::Program(const std::string &name, const Code &code)
    : Program(name, code, {}) {}
inline void Program::registerKernel(const Kernel &k) {
  if (!((NVRTC_SUCCESS) == (nvrtcAddNameExpression(_prog, k.name().c_str())))) {
    throw std::runtime_error("nvrtcAddNameExpression(_prog, k.name().c_str())");
  };
}
void Program::compile(const CompilationOptions &opt) {
  if (!((NVRTC_SUCCESS) ==
        (nvrtcCompileProgram(_prog, static_cast<int>(opt.numOptions()),
                             opt.options())))) {
    std::size_t logSize;
    nvrtcGetProgramLogSize(_prog, &logSize);
    auto log = std::string(logSize, '\0');
    nvrtcGetProgramLog(_prog, &log.front());
    throw std::runtime_error(log.c_str());
  };
}
inline std::string Program::PTX() const {
  std::size_t size = 0;
  if (!((NVRTC_SUCCESS) == (nvrtcGetPTXSize(_prog, &size)))) {
    throw std::runtime_error("nvrtcGetPTXSize(_prog, &size)");
  };
  auto str = std::string(size, '\0');
  if (!((NVRTC_SUCCESS) == (nvrtcGetPTX(_prog, &str.front())))) {
    throw std::runtime_error("nvrtcGetPTX(_prog, &str.front())");
  };
  return str;
};