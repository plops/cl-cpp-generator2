
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;

State state = {};
using namespace std::chrono_literals;
int main() {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  auto th0 = std::thread([]() -> float {
    std::this_thread::sleep_for(30ms);
    (std::cout) << ("hello ") << (std::endl);
    for (int i = 0; i < 10; (i) += (1)) {
      (std::cout) << ("push ") << (i) << (std::endl) << (std::flush);
      state._q.push_back((((1.e+0f)) * (i)));
    }
    return (2.e+0f);
  });
  auto th1 = std::thread([]() -> float {
    for (int i = 0; i < 12; (i) += (1)) {
      (std::cout) << ("                  back ") << (state._q.back())
                  << (std::endl) << (std::flush);
    }
    return (2.e+0f);
  });
  th0.join();
  th1.join();
  return 0;
};