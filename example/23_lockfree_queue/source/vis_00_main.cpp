
#include "utils.h"

#include "globals.h"

;
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

#include "rwqueue/readerwriterqueue.h"

using namespace std::chrono_literals;
State state = {};
int main(int argc, char const *const *const argv) {
  state._main_version = "8fdcc7bf51e640cbd437b110d2b8799c896b3480";
  state._code_repository =
      "https://github.com/plops/cl-cpp-generator2/tree/master/example/19_nvrtc";
  state._code_generation_time = "22:20:50 of Sunday, 2020-06-28 (GMT+1)";
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
  try {
    auto q = moodycamel::BlockingReaderWriterQueue<int>();
    auto reader = std::thread([&]() {
      auto item = int(0);
      for (auto i = 0; (i) < (100); (i) += (1)) {
        q.wait_dequeue(item);
        if (q.wait_dequeue_timed(item, std::chrono::milliseconds(5))) {
          (i)++;
        };
      };
    });
    auto writer = std::thread([&]() {
      for (auto i = 0; (i) < (100); (i) += (1)) {
        q.enqueue(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
    writer.join();
    reader.join();
  } catch (const std::exception &e) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("error") << (" ") << (std::setw(8)) << (" e.what()='")
                << (e.what()) << ("'") << (std::endl) << (std::flush);
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};