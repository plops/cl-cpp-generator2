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
    // block current thread until condition variable is notified or spurious
    // wakeup occurs. loop until queue has at least one element.

    condition_.wait(lock, [this]() { return !(queue_.empty()); });
    // remove last vector from queue

    auto msg{std::move(queue_.back())};
    queue_.pop_back();
    return msg;
  }
  bool empty() {
    std::scoped_lock lock(mutex_);
    return queue_.empty();
  }
  void send(T &&msg) {
    {
      std::scoped_lock lock(mutex_);
      queue_.push_back(std::move(msg));
    }
    // lock does not need to be held for notification

    condition_.notify_one();
  }
};