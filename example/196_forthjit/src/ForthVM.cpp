// no preamble
#include "ForthVM.h"
#include "helpers.h"
#include <algorithm>
#include <iostream>
#include <vector>

int to_status(Error error) { return static_cast<int>(error); }

std::string to_upper(std::string_view text) {
  auto upper{std::string(text)};
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [&](unsigned char value) {
                   return static_cast<char>(std::toupper(value));
                 });
  return upper;
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

ForthVM::ForthVM() {}
ForthVM::~ForthVM() {
  for (auto &&word : words_) {
    if (word.jit_result) {
      gcc_jit_result_release(word.jit_result);
    }
  }
}
void ForthVM::process_line(const std::string &line) {
  auto tokens{split_on_spaces(line)};
  auto idx{(std::size_t)0};
  if (compile_mode_) {
    idx = consume_definition_tokens(tokens, idx);
    if (tokens.size() <= idx) {
      return;
    }
  }
  while (idx < tokens.size()) {
    auto upper{to_upper(tokens[idx])};
    if (upper == ":") {
      if (tokens.size() <= idx + 1) {
        throw Error::Compile_Error;
      }
      begin_definition(tokens[(idx + 1)]);
      idx = consume_definition_tokens(tokens, idx + 2);
      if (compile_mode_) {
        return;
      }
      continue;
    } else if (upper == "VARIABLE") {
      if (tokens.size() <= idx + 1) {
        throw Error::Compile_Error;
      }
      define_variable(tokens[(idx + 1)]);
      idx += 2;
      continue;
    }
    auto start{idx};
    while (idx < tokens.size()) {
      auto current{to_upper(tokens[idx])};
      if (current == ":" || current == "VARIABLE") {
        break;
      }
      idx++;
    }
    auto segment{std::vector(tokens.begin() + start, tokens.begin() + idx)};
    if (!segment.empty()) {
      execute_segment(segment);
    }
  }
}
void ForthVM::abort_pending_definition() {
  compile_mode_ = false;
  pending_name_.clear();
  pending_tokens_.clear();
}
int ForthVM::push_literal(int value) {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  return push_raw(value);
}
int ForthVM::add() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  if (data_stack_.size() < 2) {
    return to_status(Error::Stack_Error);
  }
  auto b{data_stack_.back()};
  data_stack_.pop_back();
  auto a{data_stack_.back()};
  data_stack_.pop_back();
  data_stack_.push_back(a + b);
  return kOk;
}
int ForthVM::sub() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  if (data_stack_.size() < 2) {
    return to_status(Error::Stack_Error);
  }
  auto b{data_stack_.back()};
  data_stack_.pop_back();
  auto a{data_stack_.back()};
  data_stack_.pop_back();
  data_stack_.push_back(a - b);
  return kOk;
}
int ForthVM::mul() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  if (data_stack_.size() < 2) {
    return to_status(Error::Stack_Error);
  }
  auto b{data_stack_.back()};
  data_stack_.pop_back();
  auto a{data_stack_.back()};
  data_stack_.pop_back();
  data_stack_.push_back(a * b);
  return kOk;
}
int ForthVM::dup() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  {
    if (data_stack_.size() < 1) {
      return to_status(Error::Stack_Error);
    }
    data_stack_.push_back(data_stack_.back());
  }
  return kOk;
}
int ForthVM::drop() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  {
    if (data_stack_.size() < 1) {
      return to_status(Error::Stack_Error);
    }
    data_stack_.pop_back();
  }
  return kOk;
}
int ForthVM::swap() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  {
    if (data_stack_.size() < 2) {
      return to_status(Error::Stack_Error);
    }
    auto b{data_stack_.back()};
    data_stack_.pop_back();
    auto a{data_stack_.back()};
    data_stack_.pop_back();
    data_stack_.push_back(b);
    data_stack_.push_back(a);
  }
  return kOk;
}
int ForthVM::dot() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  {
    if (data_stack_.size() < 1) {
      return to_status(Error::Stack_Error);
    }
    std::cout << data_stack_.back() << " ";
    data_stack_.pop_back();
  }
  return kOk;
}
int ForthVM::lessthan() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  {
    if (data_stack_.size() < 2) {
      return to_status(Error::Stack_Error);
    }
    auto b{data_stack_.back()};
    data_stack_.pop_back();
    auto a{data_stack_.back()};
    data_stack_.pop_back();
    data_stack_.push_back(a < b ? ((int)1) : ((int)0));
  }
  return kOk;
}
int ForthVM::greaterthan() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  {
    if (data_stack_.size() < 2) {
      return to_status(Error::Stack_Error);
    }
    auto b{data_stack_.back()};
    data_stack_.pop_back();
    auto a{data_stack_.back()};
    data_stack_.pop_back();
    data_stack_.push_back(> (a, b) ? ((int)1) : ((int)0));
  }
  return kOk;
}
int ForthVM::equal() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  {
    if (data_stack_.size() < 2) {
      return to_status(Error::Stack_Error);
    }
    auto b{data_stack_.back()};
    data_stack_.pop_back();
    auto a{data_stack_.back()};
    data_stack_.pop_back();
    data_stack_.push_back(a == b ? ((int)1) : ((int)0));
  }
  return kOk;
}
int ForthVM::fetch() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  {
    if (data_stack_.size() < 1) {
      return to_status(Error::Stack_Error);
    }
    auto idx{data_stack_.back()};
    data_stack_.pop_back();
    if (0 <= idx && idx < (int)variables_.size()) {
      data_stack_.push_back(variables_[idx].value);
    }
  }
  return kOk;
}
int ForthVM::store() {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  {
    if (data_stack_.size() < 2) {
      return to_status(Error::Stack_Error);
    }
    auto idx{data_stack_.back()};
    data_stack_.pop_back();
    auto val{data_stack_.back()};
    data_stack_.pop_back();
    if (0 <= idx && idx < (int)variables_.size()) {
      variables_[idx].value = val;
    }
  }
  return kOk;
}
int ForthVM::call_word(int index) {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  if (index < 0 || (int)words_.size() <= index) {
    return Error::Dictionary_Full;
  }
  return (words_[index].function)(this());
}
int ForthVM::pop_condition(int *out) {
  auto status{consume_fuel()};
  if (status != kOk) {
    return status;
  }
  if (data_stack_.size() < 1) {
    return to_status(Error::Stack_Error);
  }
  *out = data_stack_.back();
  data_stack_.pop_back();
  return kOk;
}
int ForthVM::consume_fuel() {
  if (fuel_ <= 0) {
    return Error::Invalid_Fuel;
  }
  fuel_--;
  return kOk;
}
int ForthVM::push_raw(int value) {
  data_stack_.push_back(value);
  return kOk;
}
void ForthVM::begin_definition(const std::string &name) {
  compile_mode_ = true;
  pending_name_ = name;
  pending_tokens_.clear();
}
void ForthVM::define_variable(const std::string &name) {
  auto idx{(int)variables_.size()};
  variables_.push_back({name, 0});
  variable_lookup_[name] = idx;
}
std::size_t
ForthVM::consume_definition_tokens(const std::vector<std::string> &tokens,
                                   std::size_t start_index) {
  auto i{start_index};
  while (i < tokens.size()) {
    auto token{tokens[i]};
    i++;
    if (to_upper(token) == ";") {
      finish_definition();
      return i;
    }
    pending_tokens_.push_back(token);
  }
  return i;
}
std::vector<Operation>
ForthVM::parse_operations(const std::vector<std::string> &tokens,
                          ParseMode mode) const {
  auto result{parse_sequence(tokens, 0, mode)};
  if (result.stop != SequenceStop::End || result.next_index != tokens.size()) {
    throw Error::Compile_Error;
  }
  return result.operations;
}
ParseResult ForthVM::parse_sequence(const std::vector<std::string> &tokens,
                                    std::size_t index, ParseMode mode) const {
  auto result{ParseResult{}};
  result.next_index = index;
  while (result.next_index < tokens.size()) {
    auto token{tokens[result.next_index]};
    auto upper{to_upper(token)};
    if (upper == "ELSE") {
      result.stop = SequenceStop::Else;
      result.next_index++;
      return result;
    }
    if (upper == "THEN") {
      result.stop = SequenceStop::Then;
      result.next_index++;
      return result;
    }
    if (upper == "IF") {
      auto true_branch{parse_sequence(tokens, result.next_index + 1, mode)};
      if (true_branch.stop == SequenceStop::End) {
        throw Error::Compile_Error;
      }
      auto false_branch{std::vector<Operation>{}};
      if (true_branch.stop == SequenceStop::Else) {
        auto parsed_false{parse_sequence(tokens, true_branch.next_index, mode)};
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
    result.next_index++;
  }
  return result;
}
Operation ForthVM::resolve_token(const std::string &token,
                                 ParseMode mode) const {
  auto value{parse_integer(token)};
  if (value) {
    return Operation::literal(*value);
  }
  auto upper{to_upper(token)};
  if (upper == ":" || upper == ";" || upper == "VARIABLE") {
    throw Error::Compile_Error;
  }
  auto primitive{lookup_primitive(upper)};
  if (primitive) {
    return Operation::primitive_op(*primitive);
  }
  auto name{normalize_dictionary_name(token)};
  auto variable{variable_lookup_.find(name)};
  if (variable != variable_lookup_.end()) {
    return Operation::literal(variable->second);
  }
  auto word{word_lookup_.find(name)};
  if (word != word_lookup_.end()) {
    return Operation::call_word(word->second);
  }
  if (mode == ParseMode::Definition) {
    throw Error::Compile_Error;
  }
  throw Error::Unknown_Word;
}
int ForthVM::execute_operations(const std::vector<Operation> &operations) {
  for (auto &&operation : operations) {
    auto status{execute_operation(operation)};
    if (status != kOk) {
      return status;
    }
  }
  return kOk;
}
int ForthVM::execute_operation(const Operation &operation) {
  switch (operation.kind) {
  case OperationKind::Literal: {
    return push_literal(operation.value);
    break;
  };
  case OperationKind::Primitive: {
    return execute_primitive(operation.primitive);
    break;
  };
  case OperationKind::CallWord: {
    return call_word(operation.value);
    break;
  };
  case OperationKind::If: {
    auto condition{0};
    auto status{pop_condition((condition))};
    if (status != kOk) {
      return status;
    }
    if (condition != 0) {
      return execute_operations(operation.true_branch);
    } else {
      return execute_operations(operation.false_branch);
    }
    break;
  };
  }
  return to_status(Error::Compile_Error);
}
int ForthVM::execute_primitive(Primitive primitive) {
  switch (primitive) {
  case Primitive::Add: {
    return add();
    break;
  };
  case Primitive::Sub: {
    return sub();
    break;
  };
  case Primitive::Mul: {
    return mul();
    break;
  };
  case Primitive::Dup: {
    return dup();
    break;
  };
  case Primitive::Drop: {
    return drop();
    break;
  };
  case Primitive::Swap: {
    return swap();
    break;
  };
  case Primitive::Dot: {
    return;
    break;
  };
  case Primitive::LessThan: {
    return lessthan();
    break;
  };
  case Primitive::GreaterThan: {
    return greaterthan();
    break;
  };
  case Primitive::Equal: {
    return equal();
    break;
  };
  case Primitive::Fetch: {
    return fetch();
    break;
  };
  case Primitive::Store: {
    return store();
    break;
  };
  }
  return to_status(Error::Compile_Error);
}
void ForthVM::validate_definition_line(
    const std::vector<std::string> &tokens) const {
  auto if_stack{std::vector<bool>{}};
  for (auto &&token : tokens) {
    auto upper{to_upper(token)};
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
bool ForthVM::is_reserved_name(const std::string &name) const {
  auto prim{lookup_primitive(name)};
  return is_reserved_token(name) || prim.has_value();
}
void ForthVM::finish_definition() {
  auto operations{parse_operations(pending_tokens_, 0)};
  auto compiler{{JITCompiler}};
  auto symbol_name{"forth_word_" + pending_name_ + "_" +
                   std::to_string(words_.size())};
  auto result{compiler.compile_word(symbol_name, operations)};
  word_lookup_[pending_name_] = (int)words_.size();
  words_.push_back({.name = pending_name_,
                    .jit_result = result.jit_result,
                    .function = result.function});
  compile_mode_ = false;
  pending_name_.clear();
  pending_tokens_.clear();
}
void ForthVM::execute_segment(const std::vector<std::string> &tokens) {
  auto operations{parse_operations(tokens, ParseMode::Immediate)};
  auto status{execute_operations(operations)};
  if (status != kOk) {
    throw static_cast<Error>(status);
  }
}
bool ForthVM::is_dictionary_full() {
  auto MAX_DICT{1000};
  return MAX_DICT <= variables_.size() + words_.size();
}
extern "C" {

int forth_push_literal(ForthVM *vm, int value) {
  return vm->push_literal(value);
}

int forth_add(ForthVM *vm) { return vm->add(); }

int forth_sub(ForthVM *vm) { return vm->sub(); }

int forth_mul(ForthVM *vm) { return vm->mul(); }

int forth_dup(ForthVM *vm) { return vm->dup(); }

int forth_drop(ForthVM *vm) { return vm->drop(); }

int forth_swap(ForthVM *vm) { return vm->swap(); }

int forth_dot(ForthVM *vm) { return vm->execute_primitive(Primitive::Dot); }

int forth_lt(ForthVM *vm) { return vm->lessthan(); }

int forth_gt(ForthVM *vm) { return vm->greaterthan(); }

int forth_eq(ForthVM *vm) { return vm->equal(); }

int forth_fetch(ForthVM *vm) { return vm->fetch(); }

int forth_store(ForthVM *vm) { return vm->store(); }

int forth_pop_condition(ForthVM *vm, int *out_condition) {
  return vm->pop_condition(out_condition);
}

int forth_call_word(ForthVM *vm, int word_index) {
  return vm->call_word(word_index);
}
}