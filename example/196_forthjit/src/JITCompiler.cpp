// no preamble
// implementation
#include "JITCompiler.h"
#include "helpers.h"
using namespace gccjit;
JITCompiler::Result
JITCompiler::compile_word(const std::string &symbol_name,
                          const std::vector<Operation> &operations) {
  auto ctx{gccjit::context::acquire()};
  auto int_type{ctx.get_type(GCC_JIT_TYPE_INT)};
  auto vm_struct{ctx.new_opaque_struct_type("ForthVM")};
  auto vm_ptr_type{vm_struct.get_pointer()};
  auto int_ptr_type{int_type.get_pointer()};
  auto param_vm{ctx.new_param(vm_ptr_type, "vm")};
  std::vector<param> word_params{param_vm};
  auto function{ctx.new_function(GCC_JIT_FUNCTION_EXPORTED, int_type,
                                 symbol_name, word_params, 0)};
  auto declare_helper{[&](const std::string &name, std::vector<param> params) {
    return ctx.new_function(GCC_JIT_FUNCTION_IMPORTED, int_type, name, params,
                            0);
  }};
  auto make_vm_only_helper{[&](const std::string &name) {
    auto helper_vm{ctx.new_param(vm_ptr_type, "vm")};
    std::vector<param> params{{helper_vm}};
    return declare_helper(name, params);
  }};
  auto make_vm_int_helper{[&](const std::string &name) {
    auto helper_vm{ctx.new_param(vm_ptr_type, "vm")};
    auto helper_value{ctx.new_param(int_type, "value")};
    std::vector<param> params{{helper_vm, helper_value}};
    return declare_helper(name, params);
  }};
  auto helper_add{make_vm_only_helper("forth_add")};
  auto helper_sub{make_vm_only_helper("forth_sub")};
  auto helper_mul{make_vm_only_helper("forth_mul")};
  auto helper_dup{make_vm_only_helper("forth_dup")};
  auto helper_drop{make_vm_only_helper("forth_drop")};
  auto helper_swap{make_vm_only_helper("forth_swap")};
  auto helper_dot{make_vm_only_helper("forth_dot")};
  auto helper_lt{make_vm_only_helper("forth_lt")};
  auto helper_gt{make_vm_only_helper("forth_gt")};
  auto helper_eq{make_vm_only_helper("forth_eq")};
  auto helper_fetch{make_vm_only_helper("forth_fetch")};
  auto helper_store{make_vm_only_helper("forth_store")};
  auto helper_push_literal{make_vm_int_helper("forth_push_literal")};
  auto helper_call_word{make_vm_int_helper("forth_call_word")};
  auto pop_vm{ctx.new_param(vm_ptr_type, "vm")};
  auto pop_out{ctx.new_param(int_ptr_type, "out_condition")};
  std::vector<param> pop_params{{pop_vm, pop_out}};
  auto helper_pop_condition{declare_helper("forth_pop_condition", pop_params)};
  auto entry_block{function.new_block("entry")};
  auto error_block{function.new_block("error")};
  auto error_value{function.new_local(int_type, "error_value")};
  entry_block.add_assignment(error_value, ctx.zero(int_type));
  auto block_counter{0};
  auto fresh_block_name{[&](std::string_view prefix) {
    auto res{std::string{prefix}};
    res += "_";
    res += std::to_string(block_counter++);
    return res;
  }};
  auto emit_checked_call{[&](block current_block, gccjit::function helper,
                             const std::vector<rvalue> &args) -> block {
    auto ok_block{function.new_block(fresh_block_name("ok"))};
    auto mutable_args{args};
    current_block.add_assignment(error_value,
                                 ctx.new_call(helper, mutable_args));
    current_block.end_with_conditional(
        ctx.new_eq(error_value, ctx.zero(int_type)), ok_block, error_block);
    return ok_block;
  }};
  std::function<block(block, const std::vector<Operation> &)> emit_operations{
      [&](block current_block, const std::vector<Operation> &ops) -> block {
        for (const auto &operation : ops) {
          switch (operation.kind) {
          case OperationKind::Literal: {
            current_block = emit_checked_call(
                current_block, helper_push_literal,
                {param_vm, ctx.new_rvalue(int_type, operation.value)});
            break;
          };
          case OperationKind::Primitive: {
            auto helper{helper_add};
            switch (operation.primitive) {
            case Primitive::Add: {
              helper = helper_add;
              break;
            };
            case Primitive::Sub: {
              helper = helper_sub;
              break;
            };
            case Primitive::Mul: {
              helper = helper_mul;
              break;
            };
            case Primitive::Dup: {
              helper = helper_dup;
              break;
            };
            case Primitive::Drop: {
              helper = helper_drop;
              break;
            };
            case Primitive::Swap: {
              helper = helper_swap;
              break;
            };
            case Primitive::Dot: {
              helper = helper_dot;
              break;
            };
            case Primitive::LessThan: {
              helper = helper_lt;
              break;
            };
            case Primitive::GreaterThan: {
              helper = helper_gt;
              break;
            };
            case Primitive::Equal: {
              helper = helper_eq;
              break;
            };
            case Primitive::Fetch: {
              helper = helper_fetch;
              break;
            };
            case Primitive::Store: {
              helper = helper_store;
              break;
            };
            }
            current_block =
                emit_checked_call(current_block, helper, {param_vm});
            break;
          };
          case OperationKind::CallWord: {
            current_block = emit_checked_call(
                current_block, helper_call_word,
                {param_vm, ctx.new_rvalue(int_type, operation.value)});
            break;
          };
          case OperationKind::If: {
            auto condition_value{
                function.new_local(int_type, fresh_block_name("condition"))};
            current_block =
                emit_checked_call(current_block, helper_pop_condition,
                                  {param_vm, condition_value.get_address()});
            auto true_block{function.new_block(fresh_block_name("if_true"))};
            auto false_block{function.new_block(fresh_block_name("if_false"))};
            auto after_block{function.new_block(fresh_block_name("after_if"))};
            current_block.end_with_conditional(
                ctx.new_ne(condition_value, ctx.zero(int_type)), true_block,
                false_block);
            auto completed_true{
                emit_operations(true_block, operation.true_branch)};
            completed_true.end_with_jump(after_block);
            auto completed_false{
                emit_operations(false_block, operations.false_branch)};
            completed_false.end_with_jump(after_block);
            current_block = after_block;
            break;
          };
          }
        }
        return current_block;
      }};
  auto completed_entry{emit_operations(entry_block, operations)};
  completed_entry.end_with_return(ctx.zero(int_type));
  error_block.end_with_return(error_value);
  auto *jit_result{ctx.compile()};
  if (!jit_result) {
    throw Error::Compile_Error;
  }
  auto *symbol{gcc_jit_result_get_code(jit_result, symbol_name.c_str())};
  if (!symbol) {
    gcc_jit_result_release(jit_result);
    throw Error::Compile_Error;
  }
  return {.jit_result = jit_result,
          .function = reinterpret_cast<CompiledWord>(symbol)};
}