//
// Created by martin on 5/13/25.
//

#ifndef IITEM_PROCESSOR_H
#define IITEM_PROCESSOR_H
#include <cstddef>

template <typename T>
class IItemProcessor {
public:
    virtual ~IItemProcessor() = default;
    virtual void process(const T& item, std::size_t item_index) = 0;
};
#endif //IITEM_PROCESSOR_H
