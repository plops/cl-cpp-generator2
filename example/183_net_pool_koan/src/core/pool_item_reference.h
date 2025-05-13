//
// Created by martin on 5/13/25.
//

#ifndef POOL_ITEM_REFERENCE_H
#define POOL_ITEM_REFERENCE_H

#include "src/interfaces/ipool_interfaces.h" // For IPoolConsumer
#include <stdexcept>
#include <memory> // std::addressof

template <typename T>
class PoolItemReference {
public:
    PoolItemReference(IPoolConsumer<T>& consumer, std::size_t index, const T& data_item)
        : consumer_(&consumer), index_(index), data_ptr_(std::addressof(data_item)) {}

    ~PoolItemReference() {
        if (consumer_) {
            consumer_->return_item_index(index_);
        }
    }

    PoolItemReference(PoolItemReference&& other) noexcept
        : consumer_(other.consumer_), index_(other.index_), data_ptr_(other.data_ptr_) {
        other.consumer_ = nullptr;
        other.data_ptr_ = nullptr;
    }
    PoolItemReference& operator=(PoolItemReference&& other) noexcept {
        if (this != &other) {
            if (consumer_) {
                consumer_->return_item_index(index_);
            }
            consumer_ = other.consumer_;
            index_ = other.index_;
            data_ptr_ = other.data_ptr_;
            other.consumer_ = nullptr;
            other.data_ptr_ = nullptr;
        }
        return *this;
    }

    PoolItemReference(const PoolItemReference&) = delete;
    PoolItemReference& operator=(const PoolItemReference&) = delete;

    const T& get() const {
        if (!consumer_ || !data_ptr_) {
            throw std::runtime_error("Accessing moved-from or invalid PoolItemReference");
        }
        return *data_ptr_;
    }
    std::size_t index() const { return index_; }

private:
    IPoolConsumer<T>* consumer_;
    std::size_t index_;
    const T* data_ptr_;
};
#endif //POOL_ITEM_REFERENCE_H
