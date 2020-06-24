
#include "utils.h"

#include "globals.h"

;
extern State state;
#include "vis_01_cu_A_rtc_code.hpp"
#include <fstream>
#include <streambuf>
#include <string>
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
const std::string &Code::code() const { return _code; };