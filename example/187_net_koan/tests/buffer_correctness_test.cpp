//
// Created by martin on 6/11/25.
//

// tests/buffer_correctness_test.cpp
#include "gtest/gtest.h"
#include "interfaces/ibuffer.h"
#include "parser.h"     // For Parser::parse_packet
#include "test_utils.h" // For TestUtils::

#include <memory>
#include <string>
#include <vector>

// Define a parameterized test fixture
class BufferCorrectnessTest : public ::testing::TestWithParam<std::string> {
protected:
    std::unique_ptr<IBuffer> buffer;
    std::function<Parser::ParseOutput(std::vector<unsigned char>::const_iterator,
                                      std::vector<unsigned char>::const_iterator)>
            parser_fn;

    void SetUp() override {
        // For RingArrayBuffer, pass a reasonable capacity for tests.
        // Other buffers ignore the capacity argument in their IBuffer::create factory if not applicable.
        buffer    = IBuffer::create(GetParam(), 32 * 1024); // 32KB for ring buffer in tests
        parser_fn = [](auto begin, auto end) { return Parser::parse_packet(begin, end); };
    }

    void TearDown() override { buffer.reset(); }
};

TEST_P(BufferCorrectnessTest, SingleMessageWriteAndParse) {
    ASSERT_NE(buffer, nullptr);
    std::vector<unsigned char> payload  = {'h', 'e', 'l', 'l', 'o'};
    auto                       msg_data = TestUtils::create_serialized_message(1, payload);

    buffer->write(msg_data.data(), msg_data.size());
    EXPECT_EQ(buffer->readable_bytes(), msg_data.size());

    std::vector<NetworkProtocol::Message> parsed_messages;
    size_t                                num_parsed = buffer->parse_and_consume_stream(parser_fn, parsed_messages);

    EXPECT_EQ(num_parsed, 1);
    ASSERT_EQ(parsed_messages.size(), 1);
    EXPECT_EQ(parsed_messages[0].header.id, 1);
    EXPECT_EQ(parsed_messages[0].payload, payload);
    EXPECT_EQ(buffer->readable_bytes(), 0);
}

TEST_P(BufferCorrectnessTest, MultipleMessagesSequentiallyWrittenAndParsed) {
    ASSERT_NE(buffer, nullptr);
    std::vector<NetworkProtocol::Message> parsed_messages;
    size_t                                total_parsed = 0;

    std::vector<unsigned char> payload1  = {'o', 'n', 'e'};
    auto                       msg_data1 = TestUtils::create_serialized_message(10, payload1);
    buffer->write(msg_data1.data(), msg_data1.size());
    total_parsed += buffer->parse_and_consume_stream(parser_fn, parsed_messages);

    std::vector<unsigned char> payload2  = {'t', 'w', 'o'};
    auto                       msg_data2 = TestUtils::create_serialized_message(11, payload2);
    buffer->write(msg_data2.data(), msg_data2.size());
    total_parsed += buffer->parse_and_consume_stream(parser_fn, parsed_messages);

    EXPECT_EQ(total_parsed, 2); // Or check cumulative parsed_messages.size()
    ASSERT_EQ(parsed_messages.size(), 2);
    EXPECT_EQ(parsed_messages[0].header.id, 10);
    EXPECT_EQ(parsed_messages[0].payload, payload1);
    EXPECT_EQ(parsed_messages[1].header.id, 11);
    EXPECT_EQ(parsed_messages[1].payload, payload2);
    EXPECT_EQ(buffer->readable_bytes(), 0);
}

TEST_P(BufferCorrectnessTest, ChunkOfMessagesWrittenAndParsed) {
    ASSERT_NE(buffer, nullptr);
    auto chunk = TestUtils::generate_packet_chunk(5, 50, 100); // 5 messages, 50 byte avg payload
    buffer->write(chunk.data(), chunk.size());
    EXPECT_EQ(buffer->readable_bytes(), chunk.size());

    std::vector<NetworkProtocol::Message> parsed_messages;
    size_t                                num_parsed = buffer->parse_and_consume_stream(parser_fn, parsed_messages);

    EXPECT_EQ(num_parsed, 5);
    ASSERT_EQ(parsed_messages.size(), 5);
    for (size_t i = 0; i < 5; ++i) { EXPECT_EQ(parsed_messages[i].header.id, 100 + i); }
    EXPECT_EQ(buffer->readable_bytes(), 0);
}

TEST_P(BufferCorrectnessTest, PartialMessageThenCompletion) {
    ASSERT_NE(buffer, nullptr);
    std::vector<unsigned char> payload  = {'c', 'o', 'm', 'p', 'l', 'e', 't', 'e'};
    auto                       msg_data = TestUtils::create_serialized_message(200, payload);

    // Write first part (e.g., header + a bit of payload)
    size_t part1_size = NetworkProtocol::Header::SIZE + 2;
    part1_size        = std::min(part1_size, msg_data.size());
    buffer->write(msg_data.data(), part1_size);

    std::vector<NetworkProtocol::Message> parsed_messages;
    size_t                                num_parsed = buffer->parse_and_consume_stream(parser_fn, parsed_messages);
    EXPECT_EQ(num_parsed, 0); // Should need more data
    EXPECT_TRUE(parsed_messages.empty());
    EXPECT_EQ(buffer->readable_bytes(), part1_size);

    // Write the rest
    buffer->write(msg_data.data() + part1_size, msg_data.size() - part1_size);
    num_parsed = buffer->parse_and_consume_stream(parser_fn, parsed_messages);

    EXPECT_EQ(num_parsed, 1);
    ASSERT_EQ(parsed_messages.size(), 1);
    EXPECT_EQ(parsed_messages[0].header.id, 200);
    EXPECT_EQ(parsed_messages[0].payload, payload);
    EXPECT_EQ(buffer->readable_bytes(), 0);
}

TEST_P(BufferCorrectnessTest, ClearBuffer) {
    ASSERT_NE(buffer, nullptr);
    auto chunk = TestUtils::generate_packet_chunk(1, 10, 300);
    buffer->write(chunk.data(), chunk.size());
    ASSERT_GT(buffer->readable_bytes(), 0);

    buffer->clear();
    EXPECT_EQ(buffer->readable_bytes(), 0);

    std::vector<NetworkProtocol::Message> parsed_messages;
    size_t                                num_parsed = buffer->parse_and_consume_stream(parser_fn, parsed_messages);
    EXPECT_EQ(num_parsed, 0);
    EXPECT_TRUE(parsed_messages.empty());
}


// Instantiate tests for each buffer type
INSTANTIATE_TEST_SUITE_P(AllBufferTypes, BufferCorrectnessTest,
                         ::testing::Values("VectorBuffer", "DequeBuffer", "RingArrayBuffer"));

// int main(int argc, char **argv) { // GTest provides main if linked with gtest_main
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
