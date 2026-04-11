#include <algorithm>
#include <iostream>
#include <libgccjit++.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace gccjit;
enum class Error { Unknown_Word, Stack_Error, Compile_Error };
class ForthVM {
  static constexpr auto MAX_STACK = 256;
  static constexpr auto MAX_DICT = 64;
  static constexpr auto FUEL_LIMIT = 10'000;
  std::vector<int> stack;
  std::unordered_map<std::string, int> variables;
  std::unordered_map<std::string, void (*)()> dictionary;
  int fuel = 0;

public:
  void push(int val) {
    if (MAX_STACK <= stack.size()) {
      throw Error::Stack_Error;
    }
    stack.push_back(val);
  }
  int pop() {
    if (stack.empty()) {
      throw Error::Stack_Error;
    }
    auto val{stack.back()};
    stack.pop_back();
    return val;
  }
  void consume_fuel() {
    if (FUEL_LIMIT < ++fuel) {
      throw Error::Stack_Error;
    }
  }
  void dot() { std::cout << pop() << " "; }
  void dup() {
    auto v{pop()};
    push(v);
    push(v);
  }
  void drop() { pop(); }
  void swap() {
    auto a{pop()};
    auto b{pop()};
    push(a);
    push(b);
  }
};

std::string to_upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  return s;
}

void interpreter_loop() {
  auto vm{ForthVM{}};
  auto input{std::string{}};
  while (std::getline(std::cin, input)) {
    auto ss{std::stringstream(input)};
    auto token{std::string{}};
    try {
      while (ss >> token) {
        vm.consume_fuel();
        auto cmd{to_upper(token)};
      }
    }
  }
}

int main(int argc, char **argv) {
  auto ctx{gccjit::context::acquire()};
  auto int_type{ctx.get_type(GCC_JIT_TYPE_INT)};
  auto param_i{ctx.new_param(int_type, "i")};
  std::vector<gccjit::param> params = {param_i};
  auto func{ctx.new_function(GCC_JIT_FUNCTION_EXPORTED, int_type, "square",
                             params, 0)};
  auto block{func.new_block("entry")};
  auto i_rval{param_i};
  block.end_with_return(i_rval * i_rval);
  auto *result{ctx.compile()};
  if (!result) {
    std::cout << "compilation failed" << std::endl;
    return 1;
  }
  auto square{reinterpret_cast<int (*)(int)>(
      gcc_jit_result_get_code(result, "square"))};
  auto val{5};
  auto sq{square(val)};
  std::cout << "result" << " val='" << val << "' " << " sq='" << sq << "' "
            << std::endl;
  gcc_jit_result_release(result);
  return 0;
}
