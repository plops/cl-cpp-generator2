#ifndef VIS_02_TSQUEUE_H
#define VIS_02_TSQUEUE_H
#include "utils.h"
;
#include "globals.h"
;
#include <chrono>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
;
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
;
// header
template <typename T> class tsqueue {
public:
  tsqueue() = default;
  tsqueue(const tsqueue<T> &) = delete;
  virtual ~tsqueue() { clear(); }
  const T &front() {
    auto lock = std::scoped_lock(mux_deq);
    return deq.front();
  }
  const T &back() {
    auto lock = std::scoped_lock(mux_deq);
    return deq.back();
  }
  bool empty() {
    auto lock = std::scoped_lock(mux_deq);
    return deq.empty();
  }
  void clear() {
    auto lock = std::scoped_lock(mux_deq);
    return deq.clear();
  }
  T pop_front() {
    auto lock = std::scoped_lock(mux_deq);
    auto el = std::move(deq.front());
    deq.pop_front();
    return el;
  }
  T pop_back() {
    auto lock = std::scoped_lock(mux_deq);
    auto el = std::move(deq.back());
    deq.pop_back();
    return el;
  }
  void push_back(const T &item) {
    auto lock = std::scoped_lock(mux_deq);
    deq.emplace_back(std::move(item));
    auto ul = std::unique_lock<std::mutex>(mux_blocking);
    cv_blocking.notify_one();
  }
  void push_front(const T &item) {
    auto lock = std::scoped_lock(mux_deq);
    deq.emplace_front(std::move(item));
    auto ul = std::unique_lock<std::mutex>(mux_blocking);
    cv_blocking.notify_one();
  }
  void wait_while_empty() {
    auto lock = std::scoped_lock(mux_deq);
    while (empty()) {
      auto ul = std::unique_lock<std::mutex>(mux_blocking);
      cv_blocking.wait(ul);
    }
  }

protected:
  std::mutex mux_deq;
  std::mutex mux_blocking;
  std::condition_variable cv_blocking;
  std::deque<T> deq;
};
;
#endif