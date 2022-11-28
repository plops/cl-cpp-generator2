#include <chrono>
#include <experimental/mdspan>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
// https://github.com/kokkos/mdspan/wiki/A-Gentle-Introduction-to-mdspan
;
namespace stdex = std::experimental;
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
#if defined(_MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
  auto vec = std::vector<double>({1, 2, 3, 4, 5, 6, 7, 8});
  stdex::mdspan ms{vec.data(), stdex::extents{2, 2, 2}};
  auto q = ms(0, 0);
#else
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("template deduction not supported") << (" ") << (std::endl)
                << (std::flush);
  }
#endif
  return 0;
}