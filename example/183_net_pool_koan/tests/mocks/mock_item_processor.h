//
// Created by martin on 5/13/25.
//

#ifndef MOCK_ITEM_PROCESSOR_H
#define MOCK_ITEM_PROCESSOR_H
#include "src/interfaces/iitem_processor.h"
#include <gmock/gmock.h>

template <typename T>
class MockItemProcessor : public IItemProcessor<T> {
public:
    MOCK_METHOD(void, process, (const T& item, std::size_t item_index), (override));
};
#endif //MOCK_ITEM_PROCESSOR_H
