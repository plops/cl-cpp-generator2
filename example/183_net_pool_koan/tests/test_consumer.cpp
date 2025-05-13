//
// Created by martin on 5/13/25.
//
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "src/core/consumer.h"
#include "src/common/common.h"
#include "tests/mocks/mock_pool_interfaces.h"
#include "tests/mocks/mock_item_processor.h"
#include "src/core/pool_item_reference.h"

#include <stop_source>

using namespace ::testing;

template <typename T>
class ConsumerTest : public ::testing::Test {
protected:
    StrictMock<MockPoolConsumer<T>> mock_pool_consumer;
    StrictMock<MockItemProcessor<T>> mock_item_processor;
    std::stop_source stop_source_consumer;

    // Helper for creating the optional<PoolItemReference>
    std::optional<PoolItemReference<T>> make_item_ref(std::size_t index, const T& data) {
        // The PoolItemReference needs a live IPoolConsumer. We use the mock itself.
        // This ensures that when the PoolItemReference is destroyed, it calls
        // return_item_index on our mock_pool_consumer.
        return std::make_optional<PoolItemReference<T>>(
            std::ref(mock_pool_consumer), index, std::cref(data)
        );
    }
};

using ImageConsumerTest = ConsumerTest<Image>;

TEST_F(ImageConsumerTest, ProcessesItemAndReturnsIndexViaRAII) {
    Image test_data(IMAGE_SIZE_BYTES, std::byte{0xCC});
    size_t test_idx = 10;

    EXPECT_CALL(mock_pool_consumer, consume_item_ref())
        .WillOnce(Invoke([&]() { return make_item_ref(test_idx, test_data); }))
        .WillOnce(Return(std::nullopt)); // To stop the loop

    EXPECT_CALL(mock_item_processor, process(Ref(test_data), test_idx)).Times(1);
    EXPECT_CALL(mock_pool_consumer, return_item_index(test_idx)).Times(1); // Verified by RAII

    consumer_task<Image>(stop_source_consumer.get_token(), "TestImgConsumer", mock_pool_consumer, mock_item_processor);
}


TEST_F(ImageConsumerTest, StopsWhenPoolReturnsNullopt) {
    EXPECT_CALL(mock_pool_consumer, consume_item_ref()).WillOnce(Return(std::nullopt));
    // process and return_item_index should not be called.
    consumer_task<Image>(stop_source_consumer.get_token(), "TestImgConsumer", mock_pool_consumer, mock_item_processor);
}

TEST_F(ImageConsumerTest, StopsOnToken) {
    EXPECT_CALL(mock_pool_consumer, consume_item_ref())
        .WillRepeatedly(Invoke([this]() {
            if (stop_source_consumer.get_token().stop_requested()) {
                return std::optional<PoolItemReference<Image>>();
            }
            // If not stopped, keep returning nullopt to simulate an empty queue for this test
            return std::optional<PoolItemReference<Image>>();
        }));

    std::jthread consumer_thread([&]() {
        consumer_task<Image>(stop_source_consumer.get_token(), "TestImgConsumer", mock_pool_consumer, mock_item_processor);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stop_source_consumer.request_stop();
}

// Example for Metadata (struct type)
using MetadataConsumerTest = ConsumerTest<Metadata>;
TEST_F(MetadataConsumerTest, ProcessesMetadataItem) {
    Metadata test_data = {99, 7.7f};
    size_t test_idx = 11;

    EXPECT_CALL(mock_pool_consumer, consume_item_ref())
        .WillOnce(Invoke([&]() { return make_item_ref(test_idx, test_data); }))
        .WillOnce(Return(std::nullopt));

    EXPECT_CALL(mock_item_processor, process(Eq(test_data), test_idx)).Times(1); // Use Eq for struct
    EXPECT_CALL(mock_pool_consumer, return_item_index(test_idx)).Times(1);

    consumer_task<Metadata>(stop_source_consumer.get_token(), "TestMetaConsumer", mock_pool_consumer, mock_item_processor);
}