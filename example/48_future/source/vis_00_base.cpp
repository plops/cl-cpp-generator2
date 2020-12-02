
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <experimental/future>
#include <future>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;

// implementation
State state = {};
int main(int argc, char **argv) {
  state._main_version = "6732e320dde7d5ebed0c6da249887847f797b3ab";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/48_future";
  state._code_generation_time = "08:33:41 of Wednesday, 2020-12-02 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("start main") << (" ") << (std::setw(8))
                << (" state._main_version='") << (state._main_version) << ("'")
                << (std::setw(8)) << (" state._code_repository='")
                << (state._code_repository) << ("'") << (std::setw(8))
                << (" state._code_generation_time='")
                << (state._code_generation_time) << ("'") << (std::endl)
                << (std::flush);
  }
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("start") << (" ") << (std::setw(8)) << (" argc='") << (argc)
                << ("'") << (std::setw(8)) << (" argv[0]='") << (argv[0])
                << ("'") << (std::endl) << (std::flush);
  }
  auto results = std::vector<std::future<int>>();
  for (auto i = 0; (i) < (12); (i) += (1)) {
    results.push_back(([](int v) {
      auto task = std::packaged_task<int()>([&v]() {
        {

          (std::cout) << (std::setw(10))
                      << (std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count())
                      << (" ") << (std::this_thread::get_id()) << (" ")
                      << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ") << ("bla") << (" ")
                      << (std::setw(8)) << (" v='") << (v) << ("'")
                      << (std::endl) << (std::flush);
        }
        return v;
      });
      auto result = task.get_future();
      // spawn thread
      ;
      auto th = std::thread(std::move(task));
      th.detach();
      return result;
    })(i));
  }
  std::experimental::when_all(results).then([]() {
    {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("finished") << (" ") << (std::endl)
                  << (std::flush);
    }
  });
  return 0;
}