
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

#include <cxxabi.h>

using namespace std::chrono_literals;
std::string demangle(const std::string name) {
  auto status = -4;
  std::unique_ptr<char, void (*)(void *)> res{
      abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), std::free};
  if ((0) == (status)) {
    return res.get();
  } else {
    return name;
  }
}
template <class T> std::string type_name() {
  typedef typename std::remove_reference<T>::type TR;
  std::unique_ptr<char, void (*)(void *)> own(nullptr, std::free);
  std::string r = (own != nullptr) ? own.get() : typeid(TR).name();
  r = demangle(r);
  if (std::is_const<TR>::value) {
    (r) += (" const");
  }
  if (std::is_volatile<TR>::value) {
    (r) += (" volatile");
  }
  if (std::is_lvalue_reference<TR>::value) {
    (r) += ("&");
  }
  if (std::is_rvalue_reference<TR>::value) {
    (r) += ("&&");
  }
  return r;
}