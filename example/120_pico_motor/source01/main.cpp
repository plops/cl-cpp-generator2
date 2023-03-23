#include "pico/stdlib.h"
#include <iostream>

int main() {
  setup_default_uart();
  std::cout << "hello world" << std::endl;
  return 0;
}
