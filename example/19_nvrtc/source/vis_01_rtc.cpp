
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
class Code {
  const std::string _code;

public:
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
  const auto &code() const { return _code; }
};
class Header : public Code {
  const std::string _name;

public:
  template <typename... ARGS>
  Header(const std::string &name, ARGS &&... args)
      : Code(std::forward<ARGS>(args)...), _name(name) {}
  const auto &name() const { return _name; }
};
namespace detail {
template <typename... ARGS>
static inline std::vector<void *> BuildArgs(const ARGS &... args) {
  return {const_cast<void *>(reinterpret_cast<const void *>(&args))...};
}
template <typename T> class NameExtractor {
public:
  static std::string extract() {
    std::string type_name;
    nvrtcGetTypeName<T>(&type_name);
    return type_name;
  }
};
template <typename T, T y> class NameExtractor<std::integral_constant<T, y>> {
public:
  static std::string extract() { return std::to_string(y); }
};
}; // namespace detail
class Kernel {
  CUfunction _kernel = nullptr;
  std::string _name;

public:
  inline Kernel(const std::string &name) : _name(name) {}
  class TemplateParameters {
    std::string _val;
    bool _first = true;
    void addComma() {
      if (_first) {
        _first = false;
      } else {
        _val = ((_val) + (","));
      }
    }

  public:
    template <typename T> auto &addValue(const T &val) {
      addComma();
      _val = ((_val) + (std::string(val)));
      return *this;
    }
    template <typename T> auto &addType() {
      addComma();
      _val = ((_val) + (detail::NameExtractor<T>::extract()));
      return *this;
    }
    const std::string &operator()() const { return _val; };
  };
  inline Kernel &instantiate(const TemplateParameters &tp) {
    _name = ((_name) + ("<") + (tp()) + (">"));
    return *this;
  }
  template <typename... ARGS> Kernel &instantiate();
  ;
  const auto &name() const { return _name; }
};
class CompilationOptions {
  std::vector<std::string> _options;
  mutable std::vector<const char *> _chOptions;

public:
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
  CompilationOptions() = default;
  auto numOptions() const { return _options.size(); }
  const char **options() const {
    _chOptions.resize(_options.size());
    std::transform(_options.begin(), _options.end(), _chOptions.begin(),
                   [](const auto &s) { return s.c_str(); });
    return _chOptions.data();
  }
};
namespace options {
class GpuArchitecture {
  const std::string _arch;

public:
  GpuArchitecture(int major, int minor)
      : _arch(((std::string("compute_")) + (std::to_string(major)) +
               (std::to_string(minor)))) {}
  GpuArchitecture(const CudaDeviceProperties &props)
      : GpuArchitecture(props.major(), props.minor()) {}
  auto name() const { return "--gpu-architecture"; }
  auto &value() const { return _arch; }
};
enum CPPLangVer { CPP_x11, CPP_x14, CPP_x17 };
class CPPLang {
  CPPLangVer _version;

public:
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
  }
};
}; // namespace options
class Program {
  nvrtcProgram _prog;

public:
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
  Program(const std::string &name, const Code &code)
      : Program(name, code, {}) {}
  inline void registerKernel(const Kernel &k) {
    if (!((NVRTC_SUCCESS) ==
          (nvrtcAddNameExpression(_prog, k.name().c_str())))) {
      throw std::runtime_error(
          "nvrtcAddNameExpression(_prog, k.name().c_str())");
    };
  }
  void compile(const CompilationOptions &opt = {}) {
    if (!((NVRTC_SUCCESS) ==
          (nvrtcCompileProgram(_prog, static_cast<int>(opt.numOptions()),
                               opt.options())))) {
      throw std::runtime_error(
          "nvrtcCompileProgram(_prog, static_cast<int>(opt.numOptions()), "
          "opt.options())");
    };
  }
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