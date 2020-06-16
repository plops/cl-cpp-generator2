
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <algorithm>
#include <chrono>
#include <execution>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
// sudo pacman -S intel-tbb
using namespace std::chrono_literals;
State state = {};
auto get_time() { return std::chrono::high_resolution_clock::now(); }
int main() {
  state._main_version = "4ac11e942cf7c4e5bd5c86c4479fcfe867c4a226";
  state._code_repository = "http://10.1.10.5:30080/martin/py_wavelength_tune/";
  state._code_generation_time = "18:42:41 of Tuesday, 2020-06-16 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start main") << (" ")
      << (std::setw(8)) << (" state._main_version='") << (state._main_version)
      << ("'") << (std::setw(8)) << (" state._code_repository='")
      << (state._code_repository) << ("'") << (std::setw(8))
      << (" state._code_generation_time='") << (state._code_generation_time)
      << ("'") << (std::endl) << (std::flush);
  std::vector<int> v(1048576);
  std::mt19937 rng;
  std::uniform_int_distribution<int> dist(0, 255);
  rng.seed((std::random_device())());
  std::generate(begin(v), end(v), [&]() { return dist(rng); });
  auto start = get_time();
  std::sort(std::execution::par, begin(v), end(v));
  auto finish = get_time();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      ((finish) - (start)));

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("parallel run") << (" ")
      << (std::setw(8)) << (" duration.count()='") << (duration.count())
      << ("'") << (std::endl) << (std::flush);
  return 0;
};