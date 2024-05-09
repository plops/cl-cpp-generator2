#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;

void lprint(std::initializer_list<std::string> il) {
  std::chrono::duration<double> timestamp =
      std::chrono::high_resolution_clock::now() - g_start_time;
  (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
              << (std::this_thread::get_id()) << (" ");
  for (const auto &elem(type, const auto &) : il) {
    (std::cout) << (elem);
  }
  (std::cout) << (std::endl) << (std::flush);
}

int main(int argc, char **argv) {
  (g_start_time) = (std::chrono::high_resolution_clock::now());
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ", "start",
          " ", " argc='", std::to_string(argc), "'"});
  return 0;
}
