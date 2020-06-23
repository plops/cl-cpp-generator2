
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
  auto v00 = std::vector<int>(N);
  auto v01 = std::vector<int>(N);
  auto v02 = std::vector<int>(N);
  auto v03 = std::vector<int>(N);
  auto v04 = std::vector<int>(N);
  auto v05 = std::vector<int>(N);
  auto v06 = std::vector<int>(N);
  auto v07 = std::vector<int>(N);
  auto v08 = std::vector<int>(N);
  auto v09 = std::vector<int>(N);
  auto rng = std::mt19937();
  rng.seed((std::random_device())());
  auto dist = std::uniform_int_distribution<int>(0, 255);
  std::generate(begin(v00), end(v00), [&]() { return dist(rng); });
  std::generate(begin(v01), end(v01), [&]() { return dist(rng); });
  std::generate(begin(v02), end(v02), [&]() { return dist(rng); });
  std::generate(begin(v03), end(v03), [&]() { return dist(rng); });
  std::generate(begin(v04), end(v04), [&]() { return dist(rng); });
  std::generate(begin(v05), end(v05), [&]() { return dist(rng); });
  std::generate(begin(v06), end(v06), [&]() { return dist(rng); });
  std::generate(begin(v07), end(v07), [&]() { return dist(rng); });
  std::generate(begin(v08), end(v08), [&]() { return dist(rng); });
  std::generate(begin(v09), end(v09), [&]() { return dist(rng); });
  auto start = get_time();
  tbb::parallel_invoke([&]() { std::sort(begin(v00), end(v00)); },
                       [&]() { std::sort(begin(v01), end(v01)); },
                       [&]() { std::sort(begin(v02), end(v02)); },
                       [&]() { std::sort(begin(v03), end(v03)); },
                       [&]() { std::sort(begin(v04), end(v04)); },
                       [&]() { std::sort(begin(v05), end(v05)); },
                       [&]() { std::sort(begin(v06), end(v06)); },
                       [&]() { std::sort(begin(v07), end(v07)); },
                       [&]() { std::sort(begin(v08), end(v08)); },
                       [&]() { std::sort(begin(v09), end(v09)); });
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
  state._main_version = "e6021a67912b9a29a8a4bf5ecc6fdc879b66372f";
  state._code_repository =
      "https://github.com/plops/cl-cpp-generator2/tree/master/example/20_tbb";
  state._code_generation_time = "22:05:57 of Tuesday, 2020-06-23 (GMT+1)";
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