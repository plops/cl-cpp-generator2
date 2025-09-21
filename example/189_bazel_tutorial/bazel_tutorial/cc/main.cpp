#include "cc/my_lib/my_lib.hpp"
#include <iostream>

void main() {
  auto obj{MyClass()};
  obj.setValue(5);
  auto a1{std::array<float, 1>};
  auto a2{std::array<float, 2>};
  auto a3{std::array<float, 3>};
  (std::cout) << ("Value: ") << (obj.getValue()) << (std::endl);
  return 0;
}
