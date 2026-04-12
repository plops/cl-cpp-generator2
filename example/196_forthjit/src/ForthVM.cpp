// no preamble
#include "ForthVM.h"
#include <iostream>
void ForthVM::ForthVM() {
  {
  }
}
void ForthVM::~ForthVM() {
  {
    for (auto &&word : words_) {
      if (word.jit_result) {
        gcc_jit_result_release(word.jit_result);
      }
    }
  }
}
void ForthVM::execute_line(const std::string &line) {
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
        break();
      }
      idx++;
    }
    auto segment{{tokens.begin() + start, tokens.begin() + idx}};
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
    return Error::Stack_Underflow;
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
    return Error::Stack_Underflow;
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
    return Error::Stack_Underflow;
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
      return Error::Stack_Underflow;
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
      return Error::Stack_Underflow;
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
      return Error::Stack_Underflow;
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
      return Error::Stack_Underflow;
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
      return Error::Stack_Underflow;
    }
    auto b{data_stack_.back()};
    auto a{{data_stack_.pop_back();
    data_stack_.back();
  }
};
data_stack_.pop_back();
data_stack_.push_back(if (a < b) { (int)1; } else { (int)0; });
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
      return Error::Stack_Underflow;
    }
    auto b{data_stack_.back()};
    auto a{{data_stack_.pop_back();
    data_stack_.back();
  }
};
data_stack_.pop_back();
data_stack_.push_back(if (> (a, b)) { (int)1; } else { (int)0; });
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
      return Error::Stack_Underflow;
    }
    auto b{data_stack_.back()};
    auto a{{data_stack_.pop_back();
    data_stack_.back();
  }
};
data_stack_.pop_back();
data_stack_.push_back(if (a == b) { (int)1; } else { (int)0; });
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
      return Error::Stack_Underflow;
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
      return Error::Stack_Underflow;
    }
    auto idx{data_stack_.back()};
    auto val{{data_stack_.pop_back();
    data_stack_.back();
  }
};
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
    return Error::Stack_Underflow;
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
std-- size_t
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
  for (auto &&token : tokens) {
    auto upper{to_upper(token)};
    if (variable_lookup_.count(upper)) {
      push_literal(variable_lookup_[upper]);
    } else if (word_lookup_.count(upper)) {
      call_word(word_lookup_[upper]);
    } else {
      try(auto val{std::stoi(token)}; push_literal(val);
          , catch (t(, std::cerr("Unknown word: ", token, std::endl()))));
    }
  }
}
bool ForthVM::is_dictionary_full() {
  auto MAX_DICT{1000};
  return MAX_DICT <= variables_.size() + words_.size();
}