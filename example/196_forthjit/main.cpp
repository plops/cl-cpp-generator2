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

std::string normalize_dictionary_name(std::string_view text) {
  constexpr std::size_t kMaxNameLength{31};
  auto normalized{to_upper(text)};
  if (kMaxNameLength < normalized.size()) {
    normalized.resize(kMaxNameLength);
  }
  return normalized;
}

std::vector<std::string> split_on_spaces(const std::string &line) {
  auto tokens{std::vector<std::string>{}};
  auto current{std::string{}};
  for (auto ch : line) {
    if (ch == ' ') {
      if (!current.empty()) {
        tokens.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    tokens.push_back(current);
  }
  return tokens;
}

std::optional<int> parse_integer(std::string_view token) {
  if (token.empty()) {
    return std::nullopt;
  }
  auto value{0};
  auto *begin{token.data()};
  auto *end{token.data() + token.size()};
  auto [ptr, ec]{std::from_chars(begin, end, value)};
  if (ec != std::errc{} || ptr != end) {
    return std::nullopt;
  }
  return value;
}

const char *error_name(Error error) {
  switch (error) {
  case Error::Unknown_Word: {
    return "Unknown_Word";
    break;
  };
  case Error::Stack_error: {
    return "Stack_error";
    break;
  };
  case Error::Compile_Error: {
    return "Compile_Error";
    break;
  };
  }
  return "Compile_error";
}

int to_status(Error error) { return static_cast<int>(error); }

bool checked_add(int lhs, int rhs, int *result) {
  return !__builtin_add_overflow(lhs, rhs, result);
}

bool checked_sub(int lhs, int rhs, int *result) {
  return !__builtin_sub_overflow(lhs, rhs, result);
}

bool checked_mul(int lhs, int rhs, int *result) {
  return !__builtin_mul_overflow(lhs, rhs, result);
}

std::optional<Primitive> lookup_primitive(std::string_view token) {
  auto upper{to_upper(token)};
  if ("+" == upper) {
    return Primitive::Add;
  }
  if ("-" == upper) {
    return Primitive::Sub;
  }
  if ("*" == upper) {
    return Primitive::Mul;
  }
  if ("DUP" == upper) {
    return Primitive::Dup;
  }
  if ("DROP" == upper) {
    return Primitive::Drop;
  }
  if ("SWAP" == upper) {
    return Primitive::Swap;
  }
  if ("." == upper) {
    return Primitive::Dot;
  }
  if ("<" == upper) {
    return Primitive::LessThan;
  }
  if (">" == upper) {
    return Primitive::GreaterThan;
  }
  if ("=" == upper) {
    return Primitive::Equal;
  }
  if ("@" == upper) {
    return Primitive::Fetch;
  }
  if ("!" == upper) {
    return Primitive::Store;
  }
  return std::nullopt;
}

bool is_reserved_token(std::string_view token) {
  auto upper{to_upper(token)};
  return "IF" == upper || "ELSE" == upper || "THEN" == upper ||
         "VARIABLE" == upper || ":" == upper || ";" == upper;
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
