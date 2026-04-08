#include <iostream>
#include <libgccjit++.h>

int main(int argc, char **argv) {
  auto ctx{gccjit::context::acquire()};
  auto int_type{ctx.get_type(GCC_JIT_TYPE_INT)};
  auto param_i{ctx.new_param(int_type, "i")};
  std::vector<gccjit::param> params{{param_i}};
  auto func{ctx.new_function(GCC_JIT_FUNCTION_EXPORTED, int_type, "square",
                             params, 0)};
  return 0;
}
