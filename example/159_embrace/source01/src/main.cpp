#include <format>
#include <iostream>

int main(int argc, char **argv) {
  for ((decltype 100.0)i = 0; i < 1.00e+2F; i += 1) {
    std::cout << std::format({i});
  }
  return 0;
}
