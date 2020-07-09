
#include "utils.h"

#include "globals.h"

;
#include <CL/sycl.hpp>
#include <array>
#include <iostream>

int main(int argc, char const *const *const argv) {
  constexpr int n = 1024;
  auto a = std::array<int, n>();
  auto b = std::array<int, n>();
  auto c = std::array<int, n>();
  for (auto i = 0; (i) < (n); (i) += (1)) {
    a[i] = i;
    b[i] = i;
    c[i] = i;
  }
  auto platforms = sycl::platform::get_platforms();
  for (auto &p : platforms) {

    (std::cout) << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::setw(8))
                << (" p.get_info<sycl::info::platform::name>()='")
                << (p.get_info<sycl::info::platform::name>()) << ("'")
                << (std::endl) << (std::flush);
    auto devs = p.get_devices();
  };
  return 0;
};