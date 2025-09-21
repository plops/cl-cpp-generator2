#include "cc/my_lib/MyClass.h"
#include <iostream>

int main() {
  auto obj{MyClass()};
  obj.setValue(5);
  (std::cout) << ("Value: ") << (obj.getValue()) << (std::endl);
  return 0;
}
