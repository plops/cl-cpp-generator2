// no preamble
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
extern std::mutex g_stdout_mutex;
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "Charuco.h"
#include "index.h"
#include <opencv2/imgproc.hpp>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
std::mutex g_stdout_mutex;

extern "C" int main(int argc, char **argv) {
  g_start_time = std::chrono::high_resolution_clock::now();

  {
    {

      auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      std::cout << std::setw(10) << timestamp.count() << " "
                << std::this_thread::get_id() << " " << __FILE__ << ":"
                << __LINE__ << " " << __func__ << " "
                << "enter program"
                << " " << std::setw(8) << " argc='" << argc << "'"
                << std::setw(8) << " argv='" << argv << "'" << std::endl
                << std::flush;
    }
    auto ch = Charuco();
    ch.Shutdown();

    {

      auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      std::cout << std::setw(10) << timestamp.count() << " "
                << std::this_thread::get_id() << " " << __FILE__ << ":"
                << __LINE__ << " " << __func__ << " "
                << "exit program"
                << " " << std::endl
                << std::flush;
    }
    return 0;
  }
}
