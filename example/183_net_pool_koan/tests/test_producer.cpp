//
// Created by martin on 5/13/25.
//
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "src/core/producer.h"
#include "src/common/common.h"
#include "tests/mocks/mock_network_receiver.h"
#include "tests/mocks/mock_pool_interfaces.h"

#include <stop_token>
#include <cstring>
#include <arpa/inet.h> // htons
#include <thread>

using namespace ::testing;

class ProducerTest : public ::testing::Test {
protected:
    StrictMock<MockNetworkReceiver> mock_receiver;
    StrictMock<MockPoolProducer<Image>> mock_image_pool;
    StrictMock<MockPoolProducer<Metadata>> mock_meta_pool;
    StrictMock<MockPoolProducer<Measurement>> mock_meas_pool;

    Image dummy_image_buffer;
    Metadata dummy_meta_buffer;
    Measurement dummy_meas_buffer;

    std::stop_source stop_source_producer;

    ProducerTest() {
        dummy_image_buffer.resize(IMAGE_SIZE_BYTES);
    }

    std::vector<std::byte> create_packet(PacketType type, const void* payload, size_t payload_size) {
        std::vector<std::byte> packet_content;
        packet_content.push_back(static_cast<std::byte>(type));
        if (type == PacketType::Image) {
            uint16_t len_net = htons(static_cast<uint16_t>(payload_size));
            packet_content.insert(packet_content.end(), reinterpret_cast<const std::byte*>(&len_net), reinterpret_cast<const std::byte*>(&len_net) + sizeof(len_net));
        }
        if (payload && payload_size > 0) {
            packet_content.insert(packet_content.end(), static_cast<const std::byte*>(payload), static_cast<const std::byte*>(payload) + payload_size);
        }
        return packet_content;
    }
};

TEST_F(ProducerTest, ProcessesImagePacket) {
    std::vector<std::byte> image_data(IMAGE_SIZE_BYTES, std::byte{0xAB});
    auto packet = create_packet(PacketType::Image, image_data.data(), image_data.size());
    size_t DUMMY_INDEX = 1;

    EXPECT_CALL(mock_receiver, receive_packet())
        .WillOnce(Return(std::make_optional(packet)))
        .WillOnce(Return(std::nullopt)); // Stop after one packet

    EXPECT_CALL(mock_image_pool, acquire_free_index()).WillOnce(Return(DUMMY_INDEX));
    EXPECT_CALL(mock_image_pool, get_item_for_write(DUMMY_INDEX)).WillOnce(ReturnRef(dummy_image_buffer));
    EXPECT_CALL(mock_image_pool, submit_filled_index(DUMMY_INDEX));

    producer_task(stop_source_producer.get_token(), mock_receiver, mock_image_pool, mock_meta_pool, mock_meas_pool);
    ASSERT_EQ(dummy_image_buffer[0], std::byte{0xAB});
}

TEST_F(ProducerTest, ProcessesMetadataPacket) {
    Metadata meta_data = {123, 45.6f};
    auto packet = create_packet(PacketType::Metadata, &meta_data, sizeof(meta_data));
    size_t DUMMY_INDEX = 2;

    EXPECT_CALL(mock_receiver, receive_packet())
        .WillOnce(Return(std::make_optional(packet)))
        .WillOnce(Return(std::nullopt));

    EXPECT_CALL(mock_meta_pool, acquire_free_index()).WillOnce(Return(DUMMY_INDEX));
    EXPECT_CALL(mock_meta_pool, get_item_for_write(DUMMY_INDEX)).WillOnce(ReturnRef(dummy_meta_buffer));
    EXPECT_CALL(mock_meta_pool, submit_filled_index(DUMMY_INDEX));

    producer_task(stop_source_producer.get_token(), mock_receiver, mock_image_pool, mock_meta_pool, mock_meas_pool);
    ASSERT_EQ(dummy_meta_buffer.i, 123);
}

TEST_F(ProducerTest, HandlesReceiverStop) {
    EXPECT_CALL(mock_receiver, receive_packet()).WillOnce(Return(std::nullopt));
    producer_task(stop_source_producer.get_token(), mock_receiver, mock_image_pool, mock_meta_pool, mock_meas_pool);
    // No pool calls expected. Test passes if it finishes.
}

TEST_F(ProducerTest, HandlesPoolAcquireFailure) {
    std::vector<std::byte> image_data(IMAGE_SIZE_BYTES, std::byte{0xCD});
    auto packet = create_packet(PacketType::Image, image_data.data(), image_data.size());

    EXPECT_CALL(mock_receiver, receive_packet())
        .WillOnce(Return(std::make_optional(packet))) // Provide one packet
        .WillRepeatedly(Return(std::nullopt)); // Then stop

    EXPECT_CALL(mock_image_pool, acquire_free_index()).WillOnce(Return(std::nullopt)); // Simulate pool full/stopped

    producer_task(stop_source_producer.get_token(), mock_receiver, mock_image_pool, mock_meta_pool, mock_meas_pool);
    // get_item_for_write and submit_filled_index should not be called. Test passes if it finishes.
}

TEST_F(ProducerTest, StopsOnToken) {
    EXPECT_CALL(mock_receiver, receive_packet())
        .WillRepeatedly(Invoke([this]() {
            if (stop_source_producer.get_token().stop_requested()) {
                return std::optional<std::vector<std::byte>>();
            }
            // Return a dummy "unknown" packet to keep the producer busy if not stopped
            return std::make_optional(create_packet(PacketType::Unknown, nullptr, 0));
        }));
    // No pool calls needed for this specific stop test logic

    // Run in a separate thread to be able to request stop externally
    std::jthread producer_thread([&]() {
        producer_task(stop_source_producer.get_token(), mock_receiver, mock_image_pool, mock_meta_pool, mock_meas_pool);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Give producer a moment to start
    stop_source_producer.request_stop(); // Request stop
    // jthread joins automatically
}