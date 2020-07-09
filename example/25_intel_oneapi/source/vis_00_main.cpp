
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
    auto devices = p.get_devices();
    for (auto &d : devices) {

      (std::cout) << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                  << (" ") << (std::setw(8))
                  << (" d.get_info<sycl::info::device::name>()='")
                  << (d.get_info<sycl::info::device::name>()) << ("'")
                  << (std::endl) << (std::flush);
    };
  };
  auto s = sycl::default_selector();
  auto q = sycl::queue(s);
  auto a_size = sycl::range<1>(n);
  auto a_buf = sycl::buffer<int, 1>(a.data(), a.size());
  auto b_buf = sycl::buffer<int, 1>(b.data(), b.size());
  auto c_buf = sycl::buffer<int, 1>(c.data(), c.size());
  auto e = q.submit([&](sycl::handler &h) {
    auto a = a_buf.get_access<sycl::access::mode::read>(h);
    auto b = b_buf.get_access<sycl::access::mode::read>(h);
    auto c = c_buf.get_access<sycl::access::mode::write>(h);
    h.parallel_for(a_size,
                   [=](sycl::id<1> idx) { c[idx] = ((a[idx]) + (b[idx])); });
  });
  e.wait();
  {
    auto c = c_buf.get_access<sycl::access::mode::read>();

    (std::cout) << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::setw(8)) << (" c[0]='") << (c[0]) << ("'")
                << (std::setw(8)) << (" c[1]='") << (c[1]) << ("'")
                << (std::setw(8)) << (" c[((n)-(1))]='") << (c[((n) - (1))])
                << ("'") << (std::endl) << (std::flush);
  };
  return 0;
};