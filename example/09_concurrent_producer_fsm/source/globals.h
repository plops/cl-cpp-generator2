#ifndef GLOBALS_H

#define GLOBALS_H

#include <condition_variable>
#include <deque>
#include <thread>

template <int MaxLen> class FixedDeque : public std::deque<float> {
private:
  std::condition_variable filled_condition;
  std::mutex mutex;

public:
  void push_back(const float &val) {
    std::lock_guard<std::mutex> guard(this->mutex);
    if ((MaxLen) == (this->size())) {
      this->pop_front();
    };
    std::deque<float>::push_back(val);
    this->filled_condition.notify_all();
  }
  float back() {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->filled_condition.wait(lk);
    auto b = std::deque<float>::back();
    lk.unlock();
    return b;
  }
};

#include <chrono>
struct State {
  typeof(std::chrono::high_resolution_clock::now().time_since_epoch().count())
      _start_time;
  FixedDeque<10> _q;
};
typedef struct State State;

#endif
