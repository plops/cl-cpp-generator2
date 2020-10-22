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
template <typename T> class tsqueue {
public:
  tsqueue() = default;
  tsqueue(const tsqueue<T> &) = delete;
  virtual ~tsqueue();
  const T &front();
  const T &back();
  bool empty();
  void clear();
  T pop_front();
  T pop_back();
  void push_back(const T &item);
  void push_front(const T &item);
  void wait_while_empty();

protected:
  std::mutex mux_deq;
  std::mutex mux_blocking;
  std::condition_variable cv_blocking;
  std::deque<T> deq;
};
// header
;
#endif