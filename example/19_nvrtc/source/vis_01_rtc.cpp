
#include "utils.h"

#include "globals.h"

;
extern State state;
#include "vis_02_cu_device.cpp"
#include <cuda.h>
#include <fstream>
#include <nvrtc.h>
#include <streambuf>
#include <string>
// Code c{ <params> };  .. initialize
// Code c = Code::FromFile(fname);  .. load contents of file
// auto& code = c.code() .. get reference to internal string
;
template <typename... ARGS>
explicit Code(ARGS &&... args) : _code(std::forward<ARGS>(args)...) {}
static Code FromFile(const std::string &name) {
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
const auto &code() const { return _code; };
template <typename... ARGS>
Header(const std::string &name, ARGS &&... args)
    : Code(std::forward<ARGS>(args)...), _name(name) {}
const auto &name() const { return _name; };
namespace detail {
template <typename... ARGS>
static inline std::vector<void *> BuildArgs(const ARGS &... args) {
  return {const_cast<void *>(reinterpret_cast<const void *>(&args))...};
}
template <typename T> static std::string extract() {
  std::string type_name;
  nvrtcGetTypeName<T>(&type_name);
  return type_name;
};
template <typename T, T y> static std::string extract() {
  return std::to_string(y);
};
}; // namespace detail
class Program;
Module(const CudaContext &ctx, const Program &p) {
  cuModuleLoadDataEx(&_module, p.PTX().c_str(), 0, 0, 0);
}
auto module() const { return _module; };
inline Kernel(const std::string &name) : _name(name) {}
inline Kernel &instantiate(const TemplateParameters &tp) {
  _name = ((_name) + ("<") + (tp()) + (">"));
  return *this;
}
const auto &name() const { return _name; }
void init(const Module &m, const Program &p) {
  if (!((CUDA_SUCCESS) ==
        (cuModuleGetFunction(&_kernel, m.module(),
                             p.loweredName(*this).c_str())))) {
    throw std::runtime_error("cuModuleGetFunction(&_kernel, m.module(), "
                             "p.loweredName(*this).c_str())");
  };
};
void insert(const std::string &op) { _options.push_back(op); }
void insert(const std::string &name, const std::string &value) {
  if (value.empty()) {
    insert(name);
  } else {
    _options.push_back(((name) + ("=") + (value)));
  }
}
template <typename T> void insertOptions(const T &p) {
  insert(p.name(), p.value());
}
template <typename T, typename... TS>
void insertOptions(const T &p, const TS &... ts) {
  insert(p.name(), p.value());
  insertOptions(ts...);
}
template <typename... TS> CompilationOptions(TS &&... ts) {
  insertOptions(ts...);
}
auto numOptions() const { return _options.size(); }
const char **options() const {
  _chOptions.resize(_options.size());
  std::transform(_options.begin(), _options.end(), _chOptions.begin(),
                 [](const auto &s) { return s.c_str(); });
  return _chOptions.data();
};
namespace options {
GpuArchitecture(int major, int minor)
    : _arch(((std::string("compute_")) + (std::to_string(major)) +
             (std::to_string(minor)))) {}
GpuArchitecture(const CudaDeviceProperties &props)
    : GpuArchitecture(props.major(), props.minor()) {}
auto name() const { return "--gpu-architecture"; }
auto &value() const { return _arch; };
enum CPPLangVer { CPP_x11, CPP_x14, CPP_x17 };
CPPLang(CPPLangVer version) : _version(version) {}
auto name() const { return "--std"; }
auto value() const {
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
}; // namespace options
Program(const std::string &name, const Code &code,
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
Program(const std::string &name, const Code &code) : Program(name, code, {}) {}
inline void registerKernel(const Kernel &k) {
  if (!((NVRTC_SUCCESS) == (nvrtcAddNameExpression(_prog, k.name().c_str())))) {
    throw std::runtime_error("nvrtcAddNameExpression(_prog, k.name().c_str())");
  };
}
void compile(const CompilationOptions &opt = {}) {
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
inline std::string PTX() const {
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
namespace detail {
static inline void AddTypesToTemplate(Kernel::TemplateParameters &params) {}
template <typename T>
static inline void AddTypesToTemplate(Kernel::TemplateParameters &params) {
  params.addType<T>();
}
template <typename T, typename U, typename... REST>
static inline void AddTypesToTemplate(Kernel::TemplateParameters &params) {
  params.addType<T>();
  AddTypesToTemplate<U, REST...>(params);
}
}; // namespace detail
template <typename... ARGS> inline Kernel &Kernel::instantiate() {
  TemplateParameters tp;
  detail::AddTypesToTemplate<ARGS...>(tp);
  return instantiate(tp);
};