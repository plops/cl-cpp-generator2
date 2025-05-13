//
// Created by martin on 5/13/25.
//

#ifndef IPOOL_INTERFACES_H
#define IPOOL_INTERFACES_H

#include <optional>
#include <cstddef>

template <typename T> class PoolItemReference; // Forward declaration

template <typename T>
class IPoolProducer {
public:
    virtual ~IPoolProducer() = default;
    virtual std::optional<std::size_t> acquire_free_index() = 0;
    virtual T& get_item_for_write(std::size_t index) = 0;
    virtual void submit_filled_index(std::size_t index) = 0;
    virtual void stop_producing() = 0;
};

template <typename T>
class IPoolConsumer {
public:
    virtual ~IPoolConsumer() = default;
    virtual void return_item_index(std::size_t index) = 0; // Called by PoolItemReference
    virtual std::optional<PoolItemReference<T>> consume_item_ref() = 0;
    virtual void stop_consuming() = 0;
};

#endif //IPOOL_INTERFACES_H
