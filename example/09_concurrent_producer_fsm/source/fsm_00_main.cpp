
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <cstdlib>
#include <experimental/random>

State state = {};
using namespace std::chrono_literals;
std::condition_variable filled_condition;
std::mutex mutex;
int main() {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  auto th0 = std::thread([]() -> float {
    std::this_thread::sleep_for(30ms);
    (std::cout) << ("hello ") << (std::endl);
    for (int i = 0; i < 20; (i) += (1)) {
      std::this_thread::sleep_for(
          ((std::experimental::fundamentals_v2::randint(5, 300)) * (1ms)));
      (std::cout) << ("push ") << (i) << (std::endl) << (std::flush);
      state._q.push_back((((1.e+0f)) * (i)));
    }
    return (2.e+0f);
  });
  auto th1 = std::thread([]() -> float {
    for (int i = 0; i < 22; (i) += (1)) {
      auto b = state._q.back();
      (std::cout) << ("                  back=") << (b) << (" [");
      {
        std::lock_guard<std::mutex> guard(state._q.mutex);
        for (int i = 0; i < state._q.size(); (i) += (1)) {
          (std::cout) << (state._q[i]) << (" ");
        }
      };
      (std::cout) << ("]") << (std::endl) << (std::flush);
    }
    return (2.e+0f);
  });
  th0.join();
  th1.join();
  return 0;
};