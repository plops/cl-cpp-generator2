#include "ForthVM.h"
#include "helpers.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <libgccjit++.h>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
using namespace gccjit;
namespace {
using CompiledWord = int (*)(ForthVM *);

int main() {
  auto vm{ForthVM{}};
  auto line{std::string{}};
  while (std::getline(std::cin, line)) {
    switch (vm.process_line(line)) {
    case Error: {
      error();
      if ((error) == (Error::Compile_Error)) {
        vm.abort_pending_definition();
      }
      auto error_name_str{error_name(error)};
      (std::cout) << ("") << (" error_name_str='") << (error_name_str) << ("' ")
                  << (std::endl);
      break;
    };
    }
  }
  return 0;
}

omit - parens;
t;
format;
t;
tidy;
t;
}; // namespace