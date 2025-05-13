//
// Created by martin on 5/13/25.
//

#ifndef POOL_ITEM_REFERENCE_H
#define POOL_ITEM_REFERENCE_H

#include "src/interfaces/ipool_interfaces.h" // For IPoolConsumer
#include <stdexcept>
#include <memory> // std::addressof

/**
 * @brief RAII wrapper for an item consumed from a DataPool.
 * @details Ensures that the index of the consumed item is automatically returned
 *          to the pool's free list via the IPoolConsumer interface when this object
 *          goes out of scope. Provides const access to the underlying data item.
 * @tparam T The type of the data item being referenced.
 */
template <typename T>
class PoolItemReference {
public:
    /**
     * @brief Constructs a PoolItemReference.
     * @param consumer Reference to the pool consumer interface used to return the index.
     * @param index The index of the item within the pool.
     * @param data_item Constant reference to the actual data item in the pool's storage.
     */
    PoolItemReference(IPoolConsumer<T>& consumer, std::size_t index, const T& data_item)
        : consumer_(&consumer), index_(index), data_ptr_(std::addressof(data_item)) {}
    /**
     * @brief Destructor. Automatically returns the item's index to the pool via IPoolConsumer::return_item_index().
     */
    ~PoolItemReference() {
        if (consumer_) {
            consumer_->return_item_index(index_);
        }
    }

    /** @brief Move constructor. Takes ownership from another PoolItemReference. */
    PoolItemReference(PoolItemReference&& other) noexcept
        : consumer_(other.consumer_), index_(other.index_), data_ptr_(other.data_ptr_) {
        other.consumer_ = nullptr;
        other.data_ptr_ = nullptr;
    }

    /** @brief Move assignment operator. Takes ownership from another PoolItemReference. */
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

    // Non-copyable
    PoolItemReference(const PoolItemReference&) = delete;
    PoolItemReference& operator=(const PoolItemReference&) = delete;

    /**
     * @brief Gets a constant reference to the underlying data item.
     * @return Constant reference to the data item.
     * @throw std::runtime_error If the reference has been moved-from or is otherwise invalid.
     */
    const T& get() const {
        if (!consumer_ || !data_ptr_) {
            throw std::runtime_error("Accessing moved-from or invalid PoolItemReference");
        }
        return *data_ptr_;
    }

    /**
     * @brief Gets the index of the item within its pool.
     * @return The pool index.
     */
    std::size_t index() const { return index_; }

private:
    IPoolConsumer<T>* consumer_; ///< Pointer to the consumer interface for returning the index. Null if moved-from.
    std::size_t index_; ///< The index of the referenced item in the pool.
    const T* data_ptr_; ///< Pointer to the actual data item in the pool storage. Null if moved-from.
};
#endif //POOL_ITEM_REFERENCE_H
