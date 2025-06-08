//
// Created by martin on 6/8/25.
//

// tests/parser_test.cpp
#include "parser.h" // The SUT (System Under Test)
#include "gtest/gtest.h"
#include "protocol.h" // For creating test messages and constants

#include <cstdint>
#include <cstring> // For std::memcpy
#include <vector>

// Koan: The Alchemist's Bench - Helper to Forge Test Messages
// This function creates a raw byte vector representing a serialized message.
std::vector<unsigned char> forge_serialized_message(uint64_t id, uint8_t version,
                                                    const std::vector<unsigned char>& payload_data) {
    NetworkProtocol::Message test_msg;
    test_msg.header.id      = id;
    test_msg.header.version = version;
    test_msg.header.length  = payload_data.size();
    test_msg.payload        = payload_data;
    return NetworkProtocol::serializeMessage(test_msg);
}

// Koan: The Empty Scroll - Testing Behavior with No Data
TEST(ParserKoans, ParseEmptyBufferYieldsNeedMoreData) {
    std::vector<unsigned char> buffer;
    auto                       result = Parser::parse_packet(buffer.cbegin(), buffer.cend());

    EXPECT_EQ(result.status, Parser::ParseResultStatus::NEED_MORE_DATA);
    EXPECT_FALSE(result.message.has_value());
    EXPECT_EQ(result.next_data_iterator, buffer.cbegin()); // Iterator should not advance
    ASSERT_NE(result.error_message.find("Insufficient data for header"), std::string::npos);
}

// Koan: A Fragment of the Header - Testing with Insufficient Header Bytes
TEST(ParserKoans, ParsePartialHeaderYieldsNeedMoreData) {
    std::vector<unsigned char> buffer = {0x01, 0x02, 0x03, 0x04, 0x05}; // Less than Header::SIZE
    auto                       result = Parser::parse_packet(buffer.cbegin(), buffer.cend());

    EXPECT_EQ(result.status, Parser::ParseResultStatus::NEED_MORE_DATA);
    EXPECT_FALSE(result.message.has_value());
    EXPECT_EQ(result.next_data_iterator, buffer.cbegin());
}

// Koan: Header Intact, Payload Awaited - Testing Full Header, Partial Payload
TEST(ParserKoans, ParseFullHeaderPartialPayloadYieldsNeedMoreData) {
    // Header indicates payload of 10 bytes, but we only provide 5.
    NetworkProtocol::Message test_msg_template;
    test_msg_template.header.id      = 1;
    test_msg_template.header.version = NetworkProtocol::CURRENT_VERSION;
    test_msg_template.header.length  = 10; // Expect 10 bytes

    std::vector<unsigned char> partial_payload = {'h', 'e', 'l', 'l', 'o'}; // Only 5 bytes

    // Manually construct the buffer with full header but only partial payload
    std::vector<unsigned char> buffer(NetworkProtocol::Header::SIZE + partial_payload.size());
    unsigned char*             ptr = buffer.data();
    NetworkProtocol::serialize_uint64(test_msg_template.header.id, ptr);
    ptr += sizeof(uint64_t);
    NetworkProtocol::serialize_uint8(test_msg_template.header.version, ptr);
    ptr += sizeof(uint8_t);
    NetworkProtocol::serialize_uint64(test_msg_template.header.length, ptr);
    ptr += sizeof(uint64_t);
    std::memcpy(ptr, partial_payload.data(), partial_payload.size());

    auto result = Parser::parse_packet(buffer.cbegin(), buffer.cend());

    EXPECT_EQ(result.status, Parser::ParseResultStatus::NEED_MORE_DATA);
    EXPECT_FALSE(result.message.has_value());
    EXPECT_EQ(result.next_data_iterator, buffer.cbegin()); // Iterator should not advance
    ASSERT_NE(result.error_message.find("Insufficient data for promised payload"), std::string::npos);
}

// Koan: The Perfectly Formed Message - Testing a Single Complete Packet
TEST(ParserKoans, ParseSingleCompleteMessageSuccessfully) {
    uint64_t                   expected_id      = 12345;
    uint8_t                    expected_version = NetworkProtocol::CURRENT_VERSION;
    std::vector<unsigned char> expected_payload = {'t', 'e', 's', 't', ' ', 'p', 'a', 'y', 'l', 'o', 'a', 'd'};
    std::vector<unsigned char> buffer = forge_serialized_message(expected_id, expected_version, expected_payload);

    auto result = Parser::parse_packet(buffer.cbegin(), buffer.cend());

    EXPECT_EQ(result.status, Parser::ParseResultStatus::SUCCESS);
    ASSERT_TRUE(result.message.has_value()); // Ensure message is present before accessing
    EXPECT_EQ(result.message->header.id, expected_id);
    EXPECT_EQ(result.message->header.version, expected_version);
    EXPECT_EQ(result.message->header.length, expected_payload.size());
    EXPECT_EQ(result.message->payload, expected_payload);
    EXPECT_EQ(result.next_data_iterator, buffer.cend()); // Consumed the whole buffer
    EXPECT_TRUE(result.error_message.empty());
}

// Koan: Message Followed by Silence (or More) - Testing Message with Trailing Data
TEST(ParserKoans, ParseMessageWithTrailingDataAdvancesIteratorCorrectly) {
    std::vector<unsigned char> payload    = {'m', 'o', 'r', 'e'};
    std::vector<unsigned char> msg_buffer = forge_serialized_message(42, NetworkProtocol::CURRENT_VERSION, payload);
    size_t                     msg_size   = msg_buffer.size();

    std::vector<unsigned char> full_buffer = msg_buffer;
    full_buffer.push_back(0xDE); // Trailing byte 1
    full_buffer.push_back(0xAD); // Trailing byte 2
    full_buffer.push_back(0xBE); // Trailing byte 3

    auto result = Parser::parse_packet(full_buffer.cbegin(), full_buffer.cend());

    EXPECT_EQ(result.status, Parser::ParseResultStatus::SUCCESS);
    ASSERT_TRUE(result.message.has_value());
    EXPECT_EQ(result.message->header.id, 42);
    EXPECT_EQ(result.message->payload, payload);

    // next_data_iterator should point to the start of the trailing data
    ASSERT_EQ(std::distance(full_buffer.cbegin(), result.next_data_iterator), msg_size);
    EXPECT_EQ(*(result.next_data_iterator), 0xDE); // Check first trailing byte
}

// Koan: A Cascade of Messages - Testing Multiple Messages Sequentially
TEST(ParserKoans, ParseMultipleMessagesCorrectly) {
    std::vector<unsigned char> payload1   = {'f', 'i', 'r', 's', 't'};
    std::vector<unsigned char> msg1_bytes = forge_serialized_message(101, NetworkProtocol::CURRENT_VERSION, payload1);

    std::vector<unsigned char> payload2   = {'s', 'e', 'c', 'o', 'n', 'd', '!', '!'};
    std::vector<unsigned char> msg2_bytes = forge_serialized_message(102, NetworkProtocol::CURRENT_VERSION, payload2);

    std::vector<unsigned char> full_buffer = msg1_bytes;
    full_buffer.insert(full_buffer.end(), msg2_bytes.begin(), msg2_bytes.end());

    // Parse first message
    auto result1 = Parser::parse_packet(full_buffer.cbegin(), full_buffer.cend());
    EXPECT_EQ(result1.status, Parser::ParseResultStatus::SUCCESS);
    ASSERT_TRUE(result1.message.has_value());
    EXPECT_EQ(result1.message->header.id, 101);
    EXPECT_EQ(result1.message->payload, payload1);
    ASSERT_EQ(std::distance(full_buffer.cbegin(), result1.next_data_iterator), msg1_bytes.size());

    // Parse second message, starting from where the first one ended
    auto result2 = Parser::parse_packet(result1.next_data_iterator, full_buffer.cend());
    EXPECT_EQ(result2.status, Parser::ParseResultStatus::SUCCESS);
    ASSERT_TRUE(result2.message.has_value());
    EXPECT_EQ(result2.message->header.id, 102);
    EXPECT_EQ(result2.message->payload, payload2);
    EXPECT_EQ(result2.next_data_iterator, full_buffer.cend()); // Consumed rest of the buffer
}

// Koan: An Unfamiliar Dialect - Testing with an Invalid Version
TEST(ParserKoans, ParseInvalidVersionYieldsInvalidDataAndSkips) {
    uint8_t wrong_version = NetworkProtocol::CURRENT_VERSION + 1;
    if (wrong_version == 0) wrong_version = 2; // handle wrap around for test
    std::vector<unsigned char> payload = {'b', 'a', 'd', 'V'};
    std::vector<unsigned char> buffer  = forge_serialized_message(777, wrong_version, payload);

    auto result = Parser::parse_packet(buffer.cbegin(), buffer.cend());

    EXPECT_EQ(result.status, Parser::ParseResultStatus::INVALID_DATA);
    ASSERT_TRUE(result.message.has_value());                  // Parser still provides the shell
    EXPECT_EQ(result.message->header.version, wrong_version); // Verify it captured the wrong version
    EXPECT_EQ(result.message->payload, payload);              // And the payload it read
    ASSERT_NE(result.error_message.find("Unsupported protocol version"), std::string::npos);

    // It should have "consumed" the entire message based on its stated length to allow skipping
    EXPECT_EQ(result.next_data_iterator, buffer.cend());
}

// Koan: A Message of Emptiness - Testing a Zero-Length Payload
TEST(ParserKoans, ParseMessageWithZeroLengthPayload) {
    std::vector<unsigned char> empty_payload;
    std::vector<unsigned char> buffer = forge_serialized_message(0, NetworkProtocol::CURRENT_VERSION, empty_payload);

    auto result = Parser::parse_packet(buffer.cbegin(), buffer.cend());

    EXPECT_EQ(result.status, Parser::ParseResultStatus::SUCCESS);
    ASSERT_TRUE(result.message.has_value());
    EXPECT_EQ(result.message->header.id, 0);
    EXPECT_EQ(result.message->header.length, 0);
    EXPECT_TRUE(result.message->payload.empty());
    EXPECT_EQ(result.next_data_iterator, buffer.cend());
}

// Koan: The Edge of Sufficiency - Exactly Enough for Header, Expecting Payload
TEST(ParserKoans, ParseExactHeaderSizeWhenPayloadExpectedYieldsNeedMoreData) {
    // Manually construct a header that expects a payload, but provide no payload bytes.
    NetworkProtocol::Header hdr_template;
    hdr_template.id      = 99;
    hdr_template.version = NetworkProtocol::CURRENT_VERSION;
    hdr_template.length  = 1; // Expects 1 byte of payload

    std::vector<unsigned char> buffer(NetworkProtocol::Header::SIZE); // Only space for header
    unsigned char*             ptr = buffer.data();
    NetworkProtocol::serialize_uint64(hdr_template.id, ptr);
    ptr += sizeof(uint64_t);
    NetworkProtocol::serialize_uint8(hdr_template.version, ptr);
    ptr += sizeof(uint8_t);
    NetworkProtocol::serialize_uint64(hdr_template.length, ptr);
    // No payload bytes added to buffer.

    auto result = Parser::parse_packet(buffer.cbegin(), buffer.cend());

    EXPECT_EQ(result.status, Parser::ParseResultStatus::NEED_MORE_DATA);
    EXPECT_FALSE(result.message.has_value());
    EXPECT_EQ(result.next_data_iterator, buffer.cbegin());
    ASSERT_NE(result.error_message.find("Insufficient data for promised payload"), std::string::npos);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Koan: The Moment of Truth - Let the Tests Reveal Wisdom or Folly
    return RUN_ALL_TESTS();
}
