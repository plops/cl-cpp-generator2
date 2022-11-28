#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
template <typename T> class MessageQueue {
public:
  std::mutex mutex_;
  std::deque<T> queue_;
  std::condition_variable condition_;
  T receive() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this]() { return !(queue_.empty()); });
    // remove last vector from queue
    ;
    auto msg = std::move(queue_.back());
    queue_.pop_back();
    return msg;
  }
  void send(T &&msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(std::move(msg));
    condition_.notify_one();
  }
};