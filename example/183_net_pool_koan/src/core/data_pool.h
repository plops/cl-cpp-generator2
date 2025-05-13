//
// Created by martin on 5/13/25.
//

#ifndef DATA_POOL_H
#define DATA_POOL_H

#include "src/common/common.h"
#include "src/common/thread_safe_queue.h"
#include "src/interfaces/ipool_interfaces.h"
#include "pool_item_reference.h" // Full definition needed now

#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream> // For error logging
template <typename T>
class DataPool : public IPoolProducer<T>, public IPoolConsumer<T> {
public:
    DataPool(std::size_t size) : pool_size_(size) {
        if constexpr (std::is_same_v<T, Image>) {
            storage_.reserve(size);
            for (std::size_t i = 0; i < size; ++i) {
                storage_.emplace_back(IMAGE_SIZE_BYTES);
            }
        } else {
            storage_.resize(size);
        }
        std::vector<std::size_t> initial_indices(size);
        std::iota(initial_indices.begin(), initial_indices.end(), 0);
        for (std::size_t index : initial_indices) {
            free_indices_.push(index);
        }
    }

    DataPool(const DataPool&) = delete;
    DataPool& operator=(const DataPool&) = delete;

    // --- IPoolProducer<T> Implementation ---
    std::optional<std::size_t> acquire_free_index() override {
        return free_indices_.pop();
    }

    T& get_item_for_write(std::size_t index) override {
        if (index >= pool_size_) throw std::out_of_range("Pool index out of range for write");
        return storage_[index];
    }

    void submit_filled_index(std::size_t index) override {
        data_queue_.push(index);
    }

    void stop_producing() override { free_indices_.stop(); }

    // --- IPoolConsumer<T> Implementation ---
    void return_item_index(std::size_t index) override {
        if (index >= pool_size_) {
            std::cerr << "Warning: Attempt to return invalid index " << index << " to pool." << std::endl;
            return;
        }
        free_indices_.push(index);
    }

    std::optional<PoolItemReference<T>> consume_item_ref() override {
        std::optional<std::size_t> index_opt = data_queue_.pop();
        if (index_opt) {
            std::size_t idx = *index_opt;
            if (idx >= pool_size_) {
                std::cerr << "Critical Error: Consumed invalid index " << idx << " from data_queue." << std::endl;
                return std::nullopt;
            }
            return PoolItemReference<T>(*this, idx, storage_[idx]);
        }
        return std::nullopt;
    }

    void stop_consuming() override { data_queue_.stop(); }

    void stop_all() {
        stop_producing();
        stop_consuming();
    }

private:
    const std::size_t pool_size_;
    std::vector<T> storage_;
    ThreadSafeQueue<std::size_t> free_indices_;
    ThreadSafeQueue<std::size_t> data_queue_;
};
#endif //DATA_POOL_H
