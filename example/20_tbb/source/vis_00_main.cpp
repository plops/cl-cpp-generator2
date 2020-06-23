
#include "utils.h"

#include "globals.h"

;
// g++ -march=native -Ofast --std=gnu++20 vis_00_main.cpp
// -I/media/sdb4/cuda/11.0.1/include/ -L /media/sdb4/cuda/11.0.1/lib -lcudart
// -lcuda https://www.youtube.com/watch?v=OLgeKfDMcLg -- Parallel Programming:
// Intro to TBB (CoffeeBeforeArch 17 Jun 2020)
;
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <algorithm>
#include <random>
#include <tbb/parallel_invoke.h>

using namespace std::chrono_literals;
State state = {};
auto get_time() { return std::chrono::high_resolution_clock::now(); }
void run() {
  constexpr int N = 1048576;
  auto v1 = std::vector<int>(N);
  auto v2 = std::vector<int>(N);
  auto rng = std::mt19937();
  rng.seed((std::random_device())());
  auto dist = std::uniform_int_distribution<int>(0, 255);
  std::generate(begin(v1), end(v1), [&]() { return dist(rng); });
  std::generate(begin(v2), end(v2), [&]() { return dist(rng); });
  auto start = get_time();
  tbb::parallel_invoke([&]() { std::sort(begin(v1), end(v1)); },
                       [&]() { std::sort(begin(v2), end(v2)); });
  auto end = get_time();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(((end) - (start)))
          .count();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("elapsed time (us)")
      << (" ") << (std::setw(8)) << (" duration='") << (duration) << ("'")
      << (std::endl) << (std::flush);
}
int main() {
  state._main_version = "113944db02bf843605f5b1b09dbadd25deb1c862";
  state._code_repository =
      "https://github.com/plops/cl-cpp-generator2/tree/master/example/20_tbb";
  state._code_generation_time = "21:56:24 of Tuesday, 2020-06-23 (GMT+1)";
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
  run();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};