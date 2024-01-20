#include "Halide.h"
#include <format>
using namespace Halide;

int main(int argc, char **argv) {
  auto gradient{Func()};
  auto x{Var()};
  auto y{Var()};
  auto e{Expr(x + y)};
  gradient(x, y) = e;
  auto output{Buffer<int32_t>(gradient.realize({800, 600}))};
  for (auto j = 0; j < output.height(); j += 1) {
    for (auto i = 0; i < output.width(); i += 1) {
      if (output(i, j) != i + j) {
        std::cout << std::format("error i='{}' j='{}'\n", i, j);
        return -1;
      }
    }
  }
  std::cout << std::format("success\n");
  return 0;
}
