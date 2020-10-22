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
*682 * template <typename T> class tsqueue {
public:
  tsqueue() = default;
  tsqueue(const tsqueue<T> &) = delete;
  *332 * virtual ~tsqueue();
  *332 *const T &front();
  *332 *const T &back();
  *332 * bool empty();
  *332 * void clear();
  *332 * T pop_front();
  *332 * T pop_back();
  *332 * void push_back(const T &item);
  *332 * void push_front(const T &item);
  *332 * void wait_while_empty();

protected:
  std::mutex mux_deq;
  std::mutex mux_blocking;
  std::condition_variable cv_blocking;
  std::deque<T> deq;
};
// header
;
#endif