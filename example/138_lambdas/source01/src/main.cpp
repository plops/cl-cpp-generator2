#include <cmath>
#include <complex>
#include <format>
#include <iostream>
// The overload set behaves similar to typecase in Common Lisp. This code
// defines a struct called 'overload' that inherits from types provided by Ts.
// The call operators of the types can be invoked directly through this
// 'overload' struct, which it inherits. The 'overload' struct does not have a
// user-defined constructor or private members and is considered an aggregate.

// During creation the object is provided with a set of lambdas as base classes
// for the overload set, it will inherit the call operator from these lambdas.

template <typename... Ts> struct overload : Ts... {
  using Ts::operator()...;
};

int main(int argc, char **argv) {
  std::cout << std::format("main entry point argc='{}' argv[0]='{}'\n", argc,
                           argv[0]);
  // 45:00 overload sets

  overload f = {[](int8_t v) { std::cout << std::format("int8_t thingy\n"); },
                [](int16_t v) { std::cout << std::format("int16_t thingy\n"); },
                [](int32_t v) { std::cout << std::format("int32_t thingy\n"); },
                [](int64_t v) { std::cout << std::format("int64_t thingy\n"); },
                [](float v) { std::cout << std::format("float thingy\n"); },
                [](double v) { std::cout << std::format("double thingy\n"); },
                [](std::complex<int8_t> v) {
                  std::cout << std::format("std::complex<int8_t> thingy\n");
                },
                [](std::complex<int16_t> v) {
                  std::cout << std::format("std::complex<int16_t> thingy\n");
                },
                [](std::complex<int32_t> v) {
                  std::cout << std::format("std::complex<int32_t> thingy\n");
                },
                [](std::complex<int64_t> v) {
                  std::cout << std::format("std::complex<int64_t> thingy\n");
                },
                [](std::complex<float> v) {
                  std::cout << std::format("std::complex<float> thingy\n");
                },
                [](std::complex<double> v) {
                  std::cout << std::format("std::complex<double> thingy\n");
                }};
  using namespace std::complex_literals;
  f(static_cast<int8_t>(2));
  f(static_cast<int16_t>(2));
  f(static_cast<int32_t>(2));
  f(static_cast<int64_t>(2));
  f(static_cast<float>(2));
  f(static_cast<double>(2));
  f(static_cast<std::complex<int8_t>>(2, 1));
  f(static_cast<std::complex<int16_t>>(2, 1));
  f(static_cast<std::complex<int32_t>>(2, 1));
  f(static_cast<std::complex<int64_t>>(2, 1));
  f(static_cast<std::complex<float>>(2, 1));
  f(static_cast<std::complex<double>>(2, 1));
}
