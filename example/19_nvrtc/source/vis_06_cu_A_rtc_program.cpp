
#include "utils.h"

#include "globals.h"

;
extern State state;
#include <nvrtc.h>

#include "vis_01_cu_A_rtc_code.hpp"
#include "vis_04_cu_A_rtc_compilation_options.hpp"
#include "vis_05_cu_A_rtc_header.hpp"

#include "vis_06_cu_A_rtc_program.hpp"

class Kernel;

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
void Program::registerKernel(const Kernel &k) {
#include "vis_07_cu_A_rtc_kernel.hpp"
  if (!((NVRTC_SUCCESS) == (nvrtcAddNameExpression(_prog, k.name().c_str())))) {
    throw std::runtime_error("nvrtcAddNameExpression(_prog, k.name().c_str())");
  };
};