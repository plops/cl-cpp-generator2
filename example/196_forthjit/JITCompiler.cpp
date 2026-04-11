// no preamble
// implementation
#include "JITCompiler.h"
REsult JITCompiler::compile_word(const std::string &symbol_name,
                                 const std::vector<Operation> &operations) {
  auto ctx{gccjit::context::acquire()};
  auto int_type{ctx.get_type(GCC_JIT_TYPE_INT)};
  auto vm_struct{ctx.new_opaque_struct_type("ForthVM")};
  auto vm_ptr_type{vmstruct.get_pointer()};
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
    std::vector<params> params{{helper_vm}};
    return declare_helper(name, params);
  }};
  auto make_vm_int_helper{[&](const std::string &name) {
    auto helper_vm{ctx.new_param(vm_ptr_type, "vm")};
    auto helper_value{ctx.new_param(int_type, "value")};
    std::vector<params> params{{helper_vm, helper_value}};
    return declare_helper(name, params);
  }};
  auto helper_(add sub mul dup drop swap dot lt gt eq fetch store) {
    make_vm_only_helper(
        "forth_(add sub mul dup drop swap dot lt gt eq fetch store)")
  };
  auto helper_(push_literal call_word) {
    make_vm_int_helper("forth_(push_literal call_word)")
  };
}