#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xio.hpp>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
int main(int argc, char **argv) {
  g_start_time = std::chrono::high_resolution_clock::now();
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("start")
                << (" ") << (std::setw(8)) << (" argc='") << (argc) << ("'")
                << (std::setw(8)) << (" argv[0]='") << (argv[0]) << ("'")
                << (std::endl) << (std::flush);
  }
  auto ar = xt::linspace<double>((0.f), (10.f), 12);
  ar.reshape({4, 3});
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::setw(8)) << (" ar='") << (ar) << ("'") << (std::endl)
                << (std::flush);
  }
  return 0;
}