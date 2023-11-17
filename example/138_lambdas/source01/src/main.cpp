#include <cmath>
#include <complex>
#include <format>
#include <iostream>
template <typename... Ts> struct overload : Ts... {
  // operator of of all of these types so which means if you write using TS
  // operator param paren dot dot it means that whatever types we give it.   you
  // know if like the call operators of those types are going to be callable
  // directly through this overload object so it's kind of inheriting the call
  // Operator so to say overload is an aggregate, no user defined constructor,
  // no private members. elements of that aggregate are the base classes.
  // initialize the overload object with aggregate initializaton using curly
  // braces and give it a bunch of lambdas as base classes for the overload set.
  // it will inherit the call operator from them. this behaves similar to
  // typecase in common lisp

  using Ts::operator()...;
};

int main(int argc, char **argv) {
  std::cout << std::format("main entry point argc='{}' argv[0]='{}'\n", argc,
                           argv[0]);
  // 45:00 overload sets

  overload f = {[](int16_t v) { std::cout << std::format("int16_t thingy\n"); },
                [](int32_t v) { std::cout << std::format("int32_t thingy\n"); },
                [](int64_t v) { std::cout << std::format("int64_t thingy\n"); },
                [](float v) { std::cout << std::format("float thingy\n"); },
                [](double v) { std::cout << std::format("double thingy\n"); },
                [](std::complex<float> v) {
                  std::cout << std::format("std::complex<float> thingy\n");
                },
                [](std::complex<double> v) {
                  std::cout << std::format("std::complex<double> thingy\n");
                }};
  f(2);
  f(2.0F);
  f(2.0);
}
