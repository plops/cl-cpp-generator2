#include <iostream>
#include <libgccjit++.h>

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
