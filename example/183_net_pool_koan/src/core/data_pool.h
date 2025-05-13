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

/**
 * @brief Manages a pool of pre-allocated data items of type T.
 * @details Implements both IPoolProducer and IPoolConsumer interfaces, allowing
 *          producers to acquire, fill, and submit items, and consumers to consume
 *          items using an RAII approach (PoolItemReference). Uses internal thread-safe
 *          queues to manage free slot indices and ready-to-consume item indices.
 * @tparam T The type of data item stored in the pool.
 */
template <typename T>
class DataPool : public IPoolProducer<T>, public IPoolConsumer<T> {
public:
    /**
     * @brief Constructs a DataPool with a specified size.
     * @details Pre-allocates storage for `size` items. If T is Image (std::vector<std::byte>),
     *          it also pre-allocates the internal buffer of each vector to IMAGE_SIZE_BYTES.
     *          Initializes the free list queue with all indices from 0 to size-1.
     * @param size The fixed number of items the pool can hold.
     */
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

    // Non-copyable/non-movable
    DataPool(const DataPool&) = delete;
    DataPool& operator=(const DataPool&) = delete;
    DataPool(DataPool&&) = delete;
    DataPool& operator=(DataPool&&) = delete;

    // --- IPoolProducer<T> Implementation ---
    std::optional<std::size_t> acquire_free_index() override {
        return free_indices_.pop();
    }

    T& get_item_for_write(std::size_t index) override {
        if (index >= pool_size_) throw std::out_of_range("Pool index out of range for write");
        // Consider adding checks in debug mode to ensure index isn't currently "free" if possible

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
        // Consider adding checks in debug mode to ensure index was actually "in use"
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
    /**
     * @brief Stops both producer and consumer interactions with the pool.
     * @details Calls stop_producing() and stop_consuming().
     */
    void stop_all() {
        stop_producing();
        stop_consuming();
    }

private:
    const std::size_t pool_size_; ///< The total number of slots in this pool.
    std::vector<T> storage_; ///< Pre-allocated storage for the pool items.
    ThreadSafeQueue<std::size_t> free_indices_; ///< Queue holding indices of available slots for producers.
    ThreadSafeQueue<std::size_t> data_queue_; ///< Queue holding indices of filled slots ready for consumers.
};
#endif //DATA_POOL_H
