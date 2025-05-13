//
// Created by martin on 5/13/25.
//

#ifndef MOCK_POOL_INTERFACES_H
#define MOCK_POOL_INTERFACES_H

#include "src/interfaces/ipool_interfaces.h"
#include "src/core/pool_item_reference.h" // For full PoolItemReference type
#include <gmock/gmock.h>

template <typename T>
class MockPoolProducer : public IPoolProducer<T> {
public:
    MOCK_METHOD(std::optional<std::size_t>, acquire_free_index, (), (override));
    MOCK_METHOD(T&, get_item_for_write, (std::size_t index), (override));
    MOCK_METHOD(void, submit_filled_index, (std::size_t index), (override));
    MOCK_METHOD(void, stop_producing, (), (override));
};

template <typename T>
class MockPoolConsumer : public IPoolConsumer<T> {
public:
    MOCK_METHOD(void, return_item_index, (std::size_t index), (override));
    MOCK_METHOD(std::optional<PoolItemReference<T>>, consume_item_ref, (), (override));
    MOCK_METHOD(void, stop_consuming, (), (override));
};

#endif //MOCK_POOL_INTERFACES_H
