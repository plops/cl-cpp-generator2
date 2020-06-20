
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
class Program {
  nvrtcProgram _prog;

public:
  Program(const std::string &name, const Code &code)
      : Program(name, code, {}) {}
};