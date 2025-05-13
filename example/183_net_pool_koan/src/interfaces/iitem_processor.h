//
// Created by martin on 5/13/25.
//

#ifndef IITEM_PROCESSOR_H
#define IITEM_PROCESSOR_H
#include <cstddef>
/**
 * @brief Interface for components that perform the actual processing logic on a consumed data item. (Strategy Pattern)
 * @tparam T The type of data item to process.
 */
template <typename T>
class IItemProcessor {
public:
    virtual ~IItemProcessor() = default;
    /**
     * @brief Processes a single data item.
     * @param item A constant reference to the data item retrieved from the pool.
     * @param item_index The original index of the item within its pool (useful for logging or correlation).
     */
    virtual void process(const T& item, std::size_t item_index) = 0;
};
#endif //IITEM_PROCESSOR_H
