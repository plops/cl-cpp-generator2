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

  overload f = {
      [](int8_t v) { std::cout << std::format("int8_t thingy v='{}'\n", v); },
      [](int16_t v) { std::cout << std::format("int16_t thingy v='{}'\n", v); },
      [](int32_t v) { std::cout << std::format("int32_t thingy v='{}'\n", v); },
      [](int64_t v) { std::cout << std::format("int64_t thingy v='{}'\n", v); },
      [](float v) { std::cout << std::format("float thingy v='{}'\n", v); },
      [](double v) { std::cout << std::format("double thingy v='{}'\n", v); },
      [](std::complex<int8_t> v) {
        std::cout << std::format(
            "std::complex<int8_t> thingy std::real(v)='{}' std::imag(v)='{}'\n",
            std::real(v), std::imag(v));
      },
      [](std::complex<int16_t> v) {
        std::cout << std::format("std::complex<int16_t> thingy "
                                 "std::real(v)='{}' std::imag(v)='{}'\n",
                                 std::real(v), std::imag(v));
      },
      [](std::complex<int32_t> v) {
        std::cout << std::format("std::complex<int32_t> thingy "
                                 "std::real(v)='{}' std::imag(v)='{}'\n",
                                 std::real(v), std::imag(v));
      },
      [](std::complex<int64_t> v) {
        std::cout << std::format("std::complex<int64_t> thingy "
                                 "std::real(v)='{}' std::imag(v)='{}'\n",
                                 std::real(v), std::imag(v));
      },
      [](std::complex<float> v) {
        std::cout << std::format(
            "std::complex<float> thingy std::real(v)='{}' std::imag(v)='{}'\n",
            std::real(v), std::imag(v));
      },
      [](std::complex<double> v) {
        std::cout << std::format(
            "std::complex<double> thingy std::real(v)='{}' std::imag(v)='{}'\n",
            std::real(v), std::imag(v));
      }};
  using namespace std::complex_literals;
  f(int8_t(2));
  f(int16_t(2));
  f(int32_t(2));
  f(int64_t(2));
  f(float(2));
  f(double(2));
  f(std::complex<int8_t>(2, 1));
  f(std::complex<int16_t>(2, 1));
  f(std::complex<int32_t>(2, 1));
  f(std::complex<int64_t>(2, 1));
  f(std::complex<float>(2, 1));
  f(std::complex<double>(2, 1));
}
