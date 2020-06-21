
#include "utils.h"

#include "globals.h"

;
extern State state;
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
};
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
  template <typename... ARGS> inline Kernel &instantiate() {
    TemplateParameters tp;
    detail::AddTypesToTemplate<ARGS...>(tp);
    return instantiate(tp);
  }
};