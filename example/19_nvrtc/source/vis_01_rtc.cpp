
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
  const std-- string _code;

public:
  template <typename... ARGS>
  explicit Code(ARGS &&... args) : _code(std::forward<ARGS>(args)...) {}
  const auto &code() const { return _code; }
};
class Program {
  nvrtcProgram _prog;
};