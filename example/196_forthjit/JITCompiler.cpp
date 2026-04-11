// no preamble
#include "JITCompiler.h"
#include "Operation.h"
#include "helpers.h"
public:
struct Result {
  gcc_jit_result *jit_result{nullptr};
  CompiledWord function{nullptr};
};
Result JITCompiler::compile_word(const std::string &symbol - name,
                                 const std::vector<Operation> &operations) {
  auto ctx{gccjit::context::acquire()};
  auto int - type{ctx.get_type(GCC_JIT_TYPE_INT)};
  auto vm - struct {
    ctx.new_opaque_struct_type("ForthVM")
  };
  auto vm - ptr - type{vm - struct.get_pointer()};
  auto param - vm{ctx.new_param(vm - ptr - type, "vm")};
  auto word - params{std::vector<gccjit::param>{param - vm}};
  auto function{ctx.new_function(GCC_JIT_FUNCTION_EXPORTED, int - type,
                                 symbol - name, word - params, 0)};
  auto entry - block{function.new_block("entry")};
  entry - block.end_with_return(ctx.zero(int - type));
  auto jit - result{ctx.compile()};
  if (!jit - result) {
    throw Error::Compile_Error;
  }
  auto symbol{gcc_jit_result_get_code(jit - result, symbol - name.c_str())};
  if (!symbol) {
    gcc_jit_result_release(jit - result);
    throw Error::Compile_Error;
  }
  return {.jit_result = jit - result,
          .function = reinterpret_cast(CompiledWord, symbol)};
}