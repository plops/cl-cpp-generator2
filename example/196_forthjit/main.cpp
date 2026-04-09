#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
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

enum class Error : int { Unknown_Word = 1, Stack_Error = 2, Compile_Error = 3 };

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
  Store,
};

enum class OperationKind { Literal, Primitive, CallWord, If };

struct Operation {
  OperationKind kind = OperationKind::Literal;
  int value = 0;
  Primitive primitive = Primitive::Add;
  std::vector<Operation> true_branch;
  std::vector<Operation> false_branch;

  static auto literal(int value) -> Operation {
    auto op = Operation{};
    op.kind = OperationKind::Literal;
    op.value = value;
    return op;
  }

  static auto primitive_op(Primitive primitive) -> Operation {
    auto op = Operation{};
    op.kind = OperationKind::Primitive;
    op.primitive = primitive;
    return op;
  }

  static auto call_word(int word_index) -> Operation {
    auto op = Operation{};
    op.kind = OperationKind::CallWord;
    op.value = word_index;
    return op;
  }

  static auto if_op(std::vector<Operation> true_branch,
                    std::vector<Operation> false_branch) -> Operation {
    auto op = Operation{};
    op.kind = OperationKind::If;
    op.true_branch = std::move(true_branch);
    op.false_branch = std::move(false_branch);
    return op;
  }
};

enum class ParseMode { Immediate, Definition };
enum class SequenceStop { End, Else, Then };

struct ParseResult {
  std::vector<Operation> operations;
  std::size_t next_index = 0;
  SequenceStop stop = SequenceStop::End;
};

class ForthVM;
using CompiledWord = int (*)(ForthVM *);

auto to_upper(std::string_view text) -> std::string {
  auto upper = std::string{text};
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::toupper(value));
                 });
  return upper;
}

auto normalize_dictionary_name(std::string_view text) -> std::string {
  constexpr std::size_t kMaxNameLength = 31;
  auto normalized = to_upper(text);
  if (kMaxNameLength < normalized.size()) {
    normalized.resize(kMaxNameLength);
  }
  return normalized;
}

auto split_on_spaces(const std::string &line) -> std::vector<std::string> {
  auto tokens = std::vector<std::string>{};
  auto current = std::string{};
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

auto parse_integer(std::string_view token) -> std::optional<int> {
  if (token.empty()) {
    return std::nullopt;
  }
  auto value = 0;
  auto *begin = token.data();
  auto *end = token.data() + token.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::nullopt;
  }
  return value;
}

auto error_name(Error error) -> const char * {
  switch (error) {
  case Error::Unknown_Word:
    return "Unknown_Word";
  case Error::Stack_Error:
    return "Stack_Error";
  case Error::Compile_Error:
    return "Compile_Error";
  }
  return "Compile_Error";
}

auto to_status(Error error) -> int { return static_cast<int>(error); }

auto checked_add(int lhs, int rhs, int *result) -> bool {
  return !__builtin_add_overflow(lhs, rhs, result);
}

auto checked_sub(int lhs, int rhs, int *result) -> bool {
  return !__builtin_sub_overflow(lhs, rhs, result);
}

auto checked_mul(int lhs, int rhs, int *result) -> bool {
  return !__builtin_mul_overflow(lhs, rhs, result);
}

auto lookup_primitive(std::string_view token) -> std::optional<Primitive> {
  auto upper = to_upper(token);
  if (upper == "+") {
    return Primitive::Add;
  }
  if (upper == "-") {
    return Primitive::Sub;
  }
  if (upper == "*") {
    return Primitive::Mul;
  }
  if (upper == "DUP") {
    return Primitive::Dup;
  }
  if (upper == "DROP") {
    return Primitive::Drop;
  }
  if (upper == "SWAP") {
    return Primitive::Swap;
  }
  if (upper == ".") {
    return Primitive::Dot;
  }
  if (upper == "<") {
    return Primitive::LessThan;
  }
  if (upper == ">") {
    return Primitive::GreaterThan;
  }
  if (upper == "=") {
    return Primitive::Equal;
  }
  if (upper == "@") {
    return Primitive::Fetch;
  }
  if (upper == "!") {
    return Primitive::Store;
  }
  return std::nullopt;
}

auto is_reserved_token(std::string_view token) -> bool {
  auto upper = to_upper(token);
  return upper == "IF" || upper == "ELSE" || upper == "THEN" ||
         upper == "VARIABLE" || upper == ":" || upper == ";";
}

class JITCompiler {
public:
  struct Result {
    gcc_jit_result *jit_result = nullptr;
    CompiledWord function = nullptr;
  };

  auto compile_word(const std::string &symbol_name,
                    const std::vector<Operation> &operations) -> Result {
    auto ctx = gccjit::context::acquire();

    auto int_type = ctx.get_type(GCC_JIT_TYPE_INT);
    auto vm_struct = ctx.new_opaque_struct_type("ForthVM");
    auto vm_ptr_type = vm_struct.get_pointer();
    auto int_ptr_type = int_type.get_pointer();

    auto param_vm = ctx.new_param(vm_ptr_type, "vm");
    std::vector<param> word_params = {param_vm};
    auto function = ctx.new_function(GCC_JIT_FUNCTION_EXPORTED, int_type,
                                     symbol_name, word_params, 0);

    auto declare_helper = [&](const std::string &name,
                              std::vector<param> params) {
      return ctx.new_function(GCC_JIT_FUNCTION_IMPORTED, int_type, name, params,
                              0);
    };

    auto make_vm_only_helper = [&](const std::string &name) {
      auto helper_vm = ctx.new_param(vm_ptr_type, "vm");
      std::vector<param> params = {helper_vm};
      return declare_helper(name, params);
    };

    auto make_vm_int_helper = [&](const std::string &name) {
      auto helper_vm = ctx.new_param(vm_ptr_type, "vm");
      auto helper_value = ctx.new_param(int_type, "value");
      std::vector<param> params = {helper_vm, helper_value};
      return declare_helper(name, params);
    };

    auto helper_push_literal = make_vm_int_helper("forth_push_literal");
    auto helper_add = make_vm_only_helper("forth_add");
    auto helper_sub = make_vm_only_helper("forth_sub");
    auto helper_mul = make_vm_only_helper("forth_mul");
    auto helper_dup = make_vm_only_helper("forth_dup");
    auto helper_drop = make_vm_only_helper("forth_drop");
    auto helper_swap = make_vm_only_helper("forth_swap");
    auto helper_dot = make_vm_only_helper("forth_dot");
    auto helper_lt = make_vm_only_helper("forth_lt");
    auto helper_gt = make_vm_only_helper("forth_gt");
    auto helper_eq = make_vm_only_helper("forth_eq");
    auto helper_fetch = make_vm_only_helper("forth_fetch");
    auto helper_store = make_vm_only_helper("forth_store");
    auto helper_call_word = make_vm_int_helper("forth_call_word");

    auto pop_vm = ctx.new_param(vm_ptr_type, "vm");
    auto pop_out = ctx.new_param(int_ptr_type, "out_condition");
    std::vector<param> pop_params = {pop_vm, pop_out};
    auto helper_pop_condition =
        declare_helper("forth_pop_condition", pop_params);

    auto entry_block = function.new_block("entry");
    auto error_block = function.new_block("error");
    auto error_value = function.new_local(int_type, "error_value");
    entry_block.add_assignment(error_value, ctx.zero(int_type));

    auto block_counter = 0;
    auto fresh_block_name = [&](std::string_view prefix) {
      return std::string{prefix} + "_" + std::to_string(block_counter++);
    };

    auto emit_checked_call =
        [&](block current_block, gccjit::function helper,
            const std::vector<rvalue> &args) -> block {
      auto ok_block = function.new_block(fresh_block_name("ok"));
      auto mutable_args = args;
      current_block.add_assignment(error_value, ctx.new_call(helper, mutable_args));
      current_block.end_with_conditional(
          ctx.new_eq(error_value, ctx.zero(int_type)), ok_block, error_block);
      return ok_block;
    };

    std::function<block(block, const std::vector<Operation> &)> emit_operations;
    emit_operations = [&](block current_block,
                          const std::vector<Operation> &ops) -> block {
      for (const auto &operation : ops) {
        switch (operation.kind) {
        case OperationKind::Literal: {
          current_block = emit_checked_call(
              current_block, helper_push_literal,
              {param_vm, ctx.new_rvalue(int_type, operation.value)});
          break;
        }
        case OperationKind::Primitive: {
          auto helper = helper_add;
          switch (operation.primitive) {
          case Primitive::Add:
            helper = helper_add;
            break;
          case Primitive::Sub:
            helper = helper_sub;
            break;
          case Primitive::Mul:
            helper = helper_mul;
            break;
          case Primitive::Dup:
            helper = helper_dup;
            break;
          case Primitive::Drop:
            helper = helper_drop;
            break;
          case Primitive::Swap:
            helper = helper_swap;
            break;
          case Primitive::Dot:
            helper = helper_dot;
            break;
          case Primitive::LessThan:
            helper = helper_lt;
            break;
          case Primitive::GreaterThan:
            helper = helper_gt;
            break;
          case Primitive::Equal:
            helper = helper_eq;
            break;
          case Primitive::Fetch:
            helper = helper_fetch;
            break;
          case Primitive::Store:
            helper = helper_store;
            break;
          }
          current_block = emit_checked_call(current_block, helper, {param_vm});
          break;
        }
        case OperationKind::CallWord: {
          current_block = emit_checked_call(
              current_block, helper_call_word,
              {param_vm, ctx.new_rvalue(int_type, operation.value)});
          break;
        }
        case OperationKind::If: {
          auto condition_value =
              function.new_local(int_type, fresh_block_name("condition"));
          current_block = emit_checked_call(
              current_block, helper_pop_condition,
              {param_vm, condition_value.get_address()});
          auto true_block = function.new_block(fresh_block_name("if_true"));
          auto false_block = function.new_block(fresh_block_name("if_false"));
          auto after_block = function.new_block(fresh_block_name("after_if"));
          current_block.end_with_conditional(
              ctx.new_ne(condition_value, ctx.zero(int_type)), true_block,
              false_block);
          auto completed_true =
              emit_operations(true_block, operation.true_branch);
          completed_true.end_with_jump(after_block);
          auto completed_false =
              emit_operations(false_block, operation.false_branch);
          completed_false.end_with_jump(after_block);
          current_block = after_block;
          break;
        }
        }
      }
      return current_block;
    };

    auto completed_entry = emit_operations(entry_block, operations);
    completed_entry.end_with_return(ctx.zero(int_type));
    error_block.end_with_return(error_value);

    auto *jit_result = ctx.compile();
    if (!jit_result) {
      throw Error::Compile_Error;
    }
    auto *symbol = gcc_jit_result_get_code(jit_result, symbol_name.c_str());
    if (!symbol) {
      gcc_jit_result_release(jit_result);
      throw Error::Compile_Error;
    }
    return {.jit_result = jit_result,
            .function = reinterpret_cast<CompiledWord>(symbol)};
  }
};

class ForthVM {
  static constexpr std::size_t MAX_STACK = 256;
  static constexpr std::size_t MAX_DICT = 64;
  static constexpr int FUEL_LIMIT = 10'000;

  struct VariableEntry {
    std::string name;
    int value = 0;
  };

  struct WordEntry {
    std::string name;
    gcc_jit_result *jit_result = nullptr;
    CompiledWord function = nullptr;
  };

  std::array<int, MAX_STACK> stack_{};
  std::size_t stack_size_ = 0;
  std::vector<VariableEntry> variables_;
  std::unordered_map<std::string, int> variable_lookup_;
  std::vector<WordEntry> words_;
  std::unordered_map<std::string, int> word_lookup_;
  bool compiling_definition_ = false;
  std::string pending_name_;
  std::vector<std::string> pending_tokens_;

public:
  ~ForthVM() {
    for (auto &word : words_) {
      if (word.jit_result) {
        gcc_jit_result_release(word.jit_result);
        word.jit_result = nullptr;
      }
    }
  }

  auto process_line(const std::string &line) -> void {
    auto tokens = split_on_spaces(line);
    auto index = std::size_t{0};

    if (compiling_definition_) {
      index = consume_definition_tokens(tokens, index);
      if (tokens.size() <= index) {
        return;
      }
    }

    while (index < tokens.size()) {
      auto upper = to_upper(tokens[index]);
      if (upper == ":") {
        if (index + 1 >= tokens.size()) {
          throw Error::Compile_Error;
        }
        begin_definition(tokens[index + 1]);
        index = consume_definition_tokens(tokens, index + 2);
        if (compiling_definition_) {
          return;
        }
        continue;
      }

      if (upper == "VARIABLE") {
        if (index + 1 >= tokens.size()) {
          throw Error::Compile_Error;
        }
        define_variable(tokens[index + 1]);
        index += 2;
        continue;
      }

      auto start = index;
      while (index < tokens.size()) {
        auto current = to_upper(tokens[index]);
        if (current == ":" || current == "VARIABLE") {
          break;
        }
        ++index;
      }
      auto segment = std::vector<std::string>{tokens.begin() + start,
                                              tokens.begin() + index};
      if (!segment.empty()) {
        execute_segment(segment);
      }
    }
  }

  auto abort_pending_definition() -> void {
    compiling_definition_ = false;
    pending_name_.clear();
    pending_tokens_.clear();
  }

  auto push_literal(int value) -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    return push_raw(value);
  }

  auto add() -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    auto rhs = 0;
    auto lhs = 0;
    if ((status = pop_raw(rhs)) != kOk || (status = pop_raw(lhs)) != kOk) {
      return status;
    }
    auto result = 0;
    if (!checked_add(lhs, rhs, &result)) {
      return to_status(Error::Stack_Error);
    }
    return push_raw(result);
  }

  auto sub() -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    auto rhs = 0;
    auto lhs = 0;
    if ((status = pop_raw(rhs)) != kOk || (status = pop_raw(lhs)) != kOk) {
      return status;
    }
    auto result = 0;
    if (!checked_sub(lhs, rhs, &result)) {
      return to_status(Error::Stack_Error);
    }
    return push_raw(result);
  }

  auto mul() -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    auto rhs = 0;
    auto lhs = 0;
    if ((status = pop_raw(rhs)) != kOk || (status = pop_raw(lhs)) != kOk) {
      return status;
    }
    auto result = 0;
    if (!checked_mul(lhs, rhs, &result)) {
      return to_status(Error::Stack_Error);
    }
    return push_raw(result);
  }

  auto dup() -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    if (stack_size_ == 0) {
      return to_status(Error::Stack_Error);
    }
    return push_raw(stack_[stack_size_ - 1]);
  }

  auto drop() -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    auto value = 0;
    return pop_raw(value);
  }

  auto swap() -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    if (stack_size_ < 2) {
      return to_status(Error::Stack_Error);
    }
    std::swap(stack_[stack_size_ - 1], stack_[stack_size_ - 2]);
    return kOk;
  }

  auto dot() -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    auto value = 0;
    if ((status = pop_raw(value)) != kOk) {
      return status;
    }
    std::cout << value << ' ';
    std::cout.flush();
    return kOk;
  }

  auto less_than() -> int {
    return comparison([](int lhs, int rhs) { return lhs < rhs; });
  }

  auto greater_than() -> int {
    return comparison([](int lhs, int rhs) { return lhs > rhs; });
  }

  auto equal() -> int {
    return comparison([](int lhs, int rhs) { return lhs == rhs; });
  }

  auto fetch() -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    auto address = 0;
    if ((status = pop_raw(address)) != kOk) {
      return status;
    }
    if (address < 0 ||
        static_cast<std::size_t>(address) >= variables_.size()) {
      return to_status(Error::Stack_Error);
    }
    return push_raw(variables_[static_cast<std::size_t>(address)].value);
  }

  auto store() -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    auto address = 0;
    auto value = 0;
    if ((status = pop_raw(address)) != kOk || (status = pop_raw(value)) != kOk) {
      return status;
    }
    if (address < 0 ||
        static_cast<std::size_t>(address) >= variables_.size()) {
      return to_status(Error::Stack_Error);
    }
    variables_[static_cast<std::size_t>(address)].value = value;
    return kOk;
  }

  auto pop_condition(int *out_condition) -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    auto value = 0;
    if ((status = pop_raw(value)) != kOk) {
      return status;
    }
    *out_condition = (value != 0) ? 1 : 0;
    return kOk;
  }

  auto execute_word_by_index(int word_index) -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    if (word_index < 0 ||
        static_cast<std::size_t>(word_index) >= words_.size()) {
      return to_status(Error::Compile_Error);
    }
    auto function = words_[static_cast<std::size_t>(word_index)].function;
    if (!function) {
      return to_status(Error::Compile_Error);
    }
    return function(this);
  }

private:
  auto consume_definition_tokens(const std::vector<std::string> &tokens,
                                 std::size_t start_index) -> std::size_t {
    auto index = start_index;
    auto line_tokens = std::vector<std::string>{};
    while (index < tokens.size() && to_upper(tokens[index]) != ";") {
      line_tokens.push_back(tokens[index]);
      ++index;
    }
    validate_definition_line(line_tokens);
    pending_tokens_.insert(pending_tokens_.end(), line_tokens.begin(),
                           line_tokens.end());
    if (index < tokens.size()) {
      finish_definition();
      return index + 1;
    }
    return index;
  }

  auto define_variable(std::string_view raw_name) -> void {
    auto name = normalize_dictionary_name(raw_name);
    if (name.empty() || is_reserved_name(name) || parse_integer(raw_name).has_value()) {
      throw Error::Compile_Error;
    }
    if (dictionary_is_full() || word_lookup_.contains(name) ||
        variable_lookup_.contains(name)) {
      throw Error::Compile_Error;
    }
    variable_lookup_[name] = static_cast<int>(variables_.size());
    variables_.push_back({.name = name, .value = 0});
  }

  auto begin_definition(std::string_view raw_name) -> void {
    auto name = normalize_dictionary_name(raw_name);
    if (name.empty() || is_reserved_name(name) || parse_integer(raw_name).has_value()) {
      throw Error::Compile_Error;
    }
    if (dictionary_is_full() || word_lookup_.contains(name) ||
        variable_lookup_.contains(name)) {
      throw Error::Compile_Error;
    }
    compiling_definition_ = true;
    pending_name_ = name;
    pending_tokens_.clear();
  }

  auto finish_definition() -> void {
    if (pending_tokens_.empty()) {
      abort_pending_definition();
      throw Error::Compile_Error;
    }
    auto operations = parse_operations(pending_tokens_, ParseMode::Definition);
    auto compiler = JITCompiler{};
    auto symbol_name = std::string{"forth_word_"} + pending_name_ + "_" +
                       std::to_string(words_.size());
    auto result = compiler.compile_word(symbol_name, operations);
    word_lookup_[pending_name_] = static_cast<int>(words_.size());
    words_.push_back({.name = pending_name_,
                      .jit_result = result.jit_result,
                      .function = result.function});
    compiling_definition_ = false;
    pending_name_.clear();
    pending_tokens_.clear();
  }

  auto execute_segment(const std::vector<std::string> &tokens) -> void {
    auto operations = parse_operations(tokens, ParseMode::Immediate);
    auto status = execute_operations(operations);
    if (status != kOk) {
      throw static_cast<Error>(status);
    }
  }

  auto parse_operations(const std::vector<std::string> &tokens,
                        ParseMode mode) const -> std::vector<Operation> {
    auto result = parse_sequence(tokens, 0, mode);
    if (result.stop != SequenceStop::End || result.next_index != tokens.size()) {
      throw Error::Compile_Error;
    }
    return result.operations;
  }

  auto parse_sequence(const std::vector<std::string> &tokens, std::size_t index,
                      ParseMode mode) const -> ParseResult {
    auto result = ParseResult{};
    result.next_index = index;
    while (result.next_index < tokens.size()) {
      auto token = tokens[result.next_index];
      auto upper = to_upper(token);

      if (upper == "ELSE") {
        result.stop = SequenceStop::Else;
        ++result.next_index;
        return result;
      }
      if (upper == "THEN") {
        result.stop = SequenceStop::Then;
        ++result.next_index;
        return result;
      }
      if (upper == "IF") {
        auto true_branch =
            parse_sequence(tokens, result.next_index + 1, mode);
        if (true_branch.stop == SequenceStop::End) {
          throw Error::Compile_Error;
        }
        auto false_branch = std::vector<Operation>{};
        if (true_branch.stop == SequenceStop::Else) {
          auto parsed_false =
              parse_sequence(tokens, true_branch.next_index, mode);
          if (parsed_false.stop != SequenceStop::Then) {
            throw Error::Compile_Error;
          }
          false_branch = std::move(parsed_false.operations);
          result.next_index = parsed_false.next_index;
        } else {
          result.next_index = true_branch.next_index;
        }
        result.operations.push_back(Operation::if_op(
            std::move(true_branch.operations), std::move(false_branch)));
        continue;
      }

      result.operations.push_back(resolve_token(token, mode));
      ++result.next_index;
    }
    return result;
  }

  auto resolve_token(const std::string &token, ParseMode mode) const -> Operation {
    if (auto value = parse_integer(token)) {
      return Operation::literal(*value);
    }

    auto upper = to_upper(token);
    if (upper == ":" || upper == ";" || upper == "VARIABLE") {
      throw Error::Compile_Error;
    }

    if (auto primitive = lookup_primitive(upper)) {
      return Operation::primitive_op(*primitive);
    }

    auto name = normalize_dictionary_name(token);
    if (auto variable = variable_lookup_.find(name);
        variable != variable_lookup_.end()) {
      return Operation::literal(variable->second);
    }
    if (auto word = word_lookup_.find(name); word != word_lookup_.end()) {
      return Operation::call_word(word->second);
    }

    if (mode == ParseMode::Definition) {
      throw Error::Compile_Error;
    }
    throw Error::Unknown_Word;
  }

  auto execute_operations(const std::vector<Operation> &operations) -> int {
    for (const auto &operation : operations) {
      auto status = execute_operation(operation);
      if (status != kOk) {
        return status;
      }
    }
    return kOk;
  }

  auto execute_operation(const Operation &operation) -> int {
    switch (operation.kind) {
    case OperationKind::Literal:
      return push_literal(operation.value);
    case OperationKind::Primitive:
      return execute_primitive(operation.primitive);
    case OperationKind::CallWord:
      return execute_word_by_index(operation.value);
    case OperationKind::If: {
      auto condition = 0;
      auto status = pop_condition(&condition);
      if (status != kOk) {
        return status;
      }
      if (condition != 0) {
        return execute_operations(operation.true_branch);
      }
      return execute_operations(operation.false_branch);
    }
    }
    return to_status(Error::Compile_Error);
  }

  auto execute_primitive(Primitive primitive) -> int {
    switch (primitive) {
    case Primitive::Add:
      return add();
    case Primitive::Sub:
      return sub();
    case Primitive::Mul:
      return mul();
    case Primitive::Dup:
      return dup();
    case Primitive::Drop:
      return drop();
    case Primitive::Swap:
      return swap();
    case Primitive::Dot:
      return dot();
    case Primitive::LessThan:
      return less_than();
    case Primitive::GreaterThan:
      return greater_than();
    case Primitive::Equal:
      return equal();
    case Primitive::Fetch:
      return fetch();
    case Primitive::Store:
      return store();
    }
    return to_status(Error::Compile_Error);
  }

  template <typename Predicate>
  auto comparison(Predicate predicate) -> int {
    auto status = consume_fuel();
    if (status != kOk) {
      return status;
    }
    auto rhs = 0;
    auto lhs = 0;
    if ((status = pop_raw(rhs)) != kOk || (status = pop_raw(lhs)) != kOk) {
      return status;
    }
    return push_raw(predicate(lhs, rhs) ? -1 : 0);
  }

  auto push_raw(int value) -> int {
    if (MAX_STACK <= stack_size_) {
      return to_status(Error::Stack_Error);
    }
    stack_[stack_size_] = value;
    ++stack_size_;
    return kOk;
  }

  auto pop_raw(int &value) -> int {
    if (stack_size_ == 0) {
      return to_status(Error::Stack_Error);
    }
    --stack_size_;
    value = stack_[stack_size_];
    return kOk;
  }

  auto consume_fuel() -> int {
    static_assert(FUEL_LIMIT > 0);
    static_assert(MAX_STACK > 0);
    if (FUEL_LIMIT <= 0) {
      return to_status(Error::Stack_Error);
    }
    if (fuel_ >= FUEL_LIMIT) {
      return to_status(Error::Stack_Error);
    }
    ++fuel_;
    return kOk;
  }

  auto validate_definition_line(const std::vector<std::string> &tokens) const
      -> void {
    auto if_stack = std::vector<bool>{};
    for (const auto &token : tokens) {
      auto upper = to_upper(token);
      if (upper == "IF") {
        if_stack.push_back(false);
        continue;
      }
      if (upper == "ELSE") {
        if (if_stack.empty() || if_stack.back()) {
          throw Error::Compile_Error;
        }
        if_stack.back() = true;
        continue;
      }
      if (upper == "THEN") {
        if (if_stack.empty()) {
          throw Error::Compile_Error;
        }
        if_stack.pop_back();
      }
    }
    if (!if_stack.empty()) {
      throw Error::Compile_Error;
    }
  }

  auto is_reserved_name(const std::string &name) const -> bool {
    return is_reserved_token(name) || lookup_primitive(name).has_value();
  }

  auto dictionary_is_full() const -> bool {
    return MAX_DICT <= variables_.size() + words_.size();
  }

  int fuel_ = 0;
};

extern "C" auto forth_push_literal(ForthVM *vm, int value) -> int {
  return vm->push_literal(value);
}

extern "C" auto forth_add(ForthVM *vm) -> int { return vm->add(); }

extern "C" auto forth_sub(ForthVM *vm) -> int { return vm->sub(); }

extern "C" auto forth_mul(ForthVM *vm) -> int { return vm->mul(); }

extern "C" auto forth_dup(ForthVM *vm) -> int { return vm->dup(); }

extern "C" auto forth_drop(ForthVM *vm) -> int { return vm->drop(); }

extern "C" auto forth_swap(ForthVM *vm) -> int { return vm->swap(); }

extern "C" auto forth_dot(ForthVM *vm) -> int { return vm->dot(); }

extern "C" auto forth_lt(ForthVM *vm) -> int { return vm->less_than(); }

extern "C" auto forth_gt(ForthVM *vm) -> int { return vm->greater_than(); }

extern "C" auto forth_eq(ForthVM *vm) -> int { return vm->equal(); }

extern "C" auto forth_fetch(ForthVM *vm) -> int { return vm->fetch(); }

extern "C" auto forth_store(ForthVM *vm) -> int { return vm->store(); }

extern "C" auto forth_pop_condition(ForthVM *vm, int *out_condition) -> int {
  return vm->pop_condition(out_condition);
}

extern "C" auto forth_call_word(ForthVM *vm, int word_index) -> int {
  return vm->execute_word_by_index(word_index);
}

} // namespace

int main() {
  auto vm = ForthVM{};
  auto line = std::string{};

  while (std::getline(std::cin, line)) {
    try {
      vm.process_line(line);
    } catch (Error error) {
      if (error == Error::Compile_Error) {
        vm.abort_pending_definition();
      }
      std::cerr << error_name(error) << '\n';
    }
  }

  return 0;
}
