//
// Created by martin on 5/13/25.
//

#ifndef IPOOL_INTERFACES_H
#define IPOOL_INTERFACES_H

#include <optional>
#include <cstddef>

template <typename T> class PoolItemReference; // Forward declaration

/**
 * @brief Interface for the producer-side interaction with a data pool.
 * @details Defines how producers acquire empty slots, write data into them, and submit them for consumption.
 * @tparam T The type of data item stored in the pool.
 */
template <typename T>
class IPoolProducer {
public:
    virtual ~IPoolProducer() = default;
    /**
     * @brief Acquires the index of a free slot from the pool.
     * @details Blocks if no free slots are currently available until one is returned or the pool is stopped.
     * @return An std::optional containing the index of the acquired slot, or std::nullopt if the pool was stopped.
     */
    virtual std::optional<std::size_t> acquire_free_index() = 0;

    /**
     * @brief Gets a mutable reference to the data item at the specified index.
     * @warning This should only be called with an index previously obtained via `acquire_free_index()` and not yet submitted.
     *          Behavior is undefined if the index is invalid or refers to a slot not currently exclusively held by the caller.
     * @param index The index of the pool slot to access.
     * @return A mutable reference to the data item.
     * @throw std::out_of_range If the index is invalid (optional check by implementation).
     */
    virtual T& get_item_for_write(std::size_t index) = 0;

    /**
     * @brief Submits a filled slot (identified by its index) to the consumer queue.
     * @warning This should only be called with an index previously obtained via `acquire_free_index()` and subsequently filled.
     * @param index The index of the slot containing data ready for consumption.
     */
    virtual void submit_filled_index(std::size_t index) = 0;

    /**
     * @brief Stops the producer-side interaction with the pool.
     * @details Typically signals the free list queue to stop, causing `acquire_free_index` to unblock and return std::nullopt.
     */
    virtual void stop_producing() = 0;
};

/**
 * @brief Interface for the consumer-side interaction with a data pool.
 * @details Defines how consumers retrieve filled items (via PoolItemReference) and how indices are returned (implicitly by PoolItemReference).
 * @tparam T The type of data item stored in the pool.
 */
template <typename T>
class IPoolConsumer {
public:
    virtual ~IPoolConsumer() = default;
    /**
     * @brief Returns a previously consumed item's index back to the pool's free list.
     * @note This method is primarily intended to be called by the PoolItemReference destructor using RAII.
     * @param index The index of the slot to return.
     */
    virtual void return_item_index(std::size_t index) = 0; // Called by PoolItemReference

    /**
     * @brief Consumes the next available filled item from the pool.
     * @details Blocks if no filled items are currently available until one is submitted or the pool is stopped.
     * @return An std::optional containing a PoolItemReference to the consumed item. The PoolItemReference provides
     *         access to the item's data and ensures its index is returned to the pool via RAII when the reference goes out of scope.
     *         Returns std::nullopt if the pool was stopped and no more items are available.
     */
    virtual std::optional<PoolItemReference<T>> consume_item_ref() = 0;

    /**
     * @brief Stops the consumer-side interaction with the pool.
     * @details Typically signals the data queue to stop, causing `consume_item_ref` to unblock and return std::nullopt.
     */
    virtual void stop_consuming() = 0;
};

#endif //IPOOL_INTERFACES_H
