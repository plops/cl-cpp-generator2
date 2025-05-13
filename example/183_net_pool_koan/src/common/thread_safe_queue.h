//
// Created by martin on 5/13/25.
//

#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <atomic>

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() = default;

    // Non-copyable and non-movable for simplicity
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue(ThreadSafeQueue&&) = delete;
    ThreadSafeQueue& operator=(ThreadSafeQueue&&) = delete;

    void push(T value) {
        {
            std::lock_guard lock(mtx_);
            // Optional: Add check here if implementing a bounded queue
            queue_.push(std::move(value));
        } // Lock released before notification
        cv_.notify_one();
    }

    // Blocking pop
    std::optional<T> pop() {
        std::unique_lock lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });

        if (stopped_ && queue_.empty()) {
            return std::nullopt; // Signal shutdown
        }

        if (queue_.empty()) {
             // This case should ideally not happen often with the wait condition,
             // but can occur due to spurious wakeups or race if stop is called
             // after wait but before checking queue_.empty() again inside the lock.
            return std::nullopt;
        }


        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

     // Non-blocking pop
    std::optional<T> try_pop() {
        std::lock_guard lock(mtx_);
        if (queue_.empty() || stopped_) { // Don't allow pops after stop signal if empty
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    bool empty() const {
        std::lock_guard lock(mtx_);
        return queue_.empty();
    }

    void stop() {
        {
            std::lock_guard lock(mtx_);
            stopped_ = true;
        } // Lock released before notification
        cv_.notify_all(); // Wake up all waiting threads
    }

private:
    mutable std::mutex mtx_; // Mutable to allow locking in const empty()
    std::queue<T> queue_;
    std::condition_variable cv_;
    std::atomic<bool> stopped_{false}; // Use atomic for safe checking without lock in some scenarios, though wait requires lock
};

#endif // THREAD_SAFE_QUEUE_H