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
constexpr auto kOk = 0;
enum class Error - int{Unknown_Word = 1, Stack_Error = 2, Compile_Error = 3};
enum class Primitive {
  Add,
  Sub,
  Mul,
  Dup,
  Drop,
  Swap,
  Dot,
  LessThan,
  GreaterThan,
  Equal,
  Fetch,
  Store
};
enum class OperationKind { Literal, Primitive, CallWord, If };
enum class ParseMode { Immediate, Definition };
enum class SequenceStop { End, Else, Then };
class ForthVM;
using CompiledWord = int (*)(ForthVM *);

std::string to_upper(std::string_view text) {
  auto upper{std::string(text)};
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [&](unsigned char value) {
                   return static_cast<char>(std::toupper(value));
                 });
  return upper;
}

}; // namespace

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
