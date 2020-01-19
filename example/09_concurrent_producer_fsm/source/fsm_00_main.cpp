
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <condition_variable>
#include <deque>
#include <thread>

State state = {};
int main() {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  auto th = std::thread([]()
                        : float {
                          (std::cout) << ("hello ") << (std::endl);
                          return (2.e+0f);
                        });
  th.join();
  return 0;
};