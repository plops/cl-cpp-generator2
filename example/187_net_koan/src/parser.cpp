//
// Created by martin on 6/8/25.
//

// src/parser.cpp
#include "parser.h"
#include <iostream> // For diagnostic messages

namespace Parser {

    ParseOutput parse_packet(std::vector<unsigned char>::const_iterator begin,
                             std::vector<unsigned char>::const_iterator end) {
        // Koan: Measuring the Stream - How much data do we have to work with?
        const auto available_data_size = static_cast<size_t>(std::distance(begin, end));

        // Koan: The First Checkpoint - The Header's Presence
        // Can we even read the fixed-size header?
        if (available_data_size < NetworkProtocol::Header::SIZE) {
            return {std::nullopt, begin, ParseResultStatus::NEED_MORE_DATA, "Insufficient data for header."};
        }

        // Koan: Unveiling the Header - Reading its Fields
        // We have enough for a header, let's deserialize it.
        // We advance a raw pointer for easy deserialization from the iterator's data.
        const unsigned char*    current_pos = &(*begin);
        NetworkProtocol::Header hdr;

        hdr.id = NetworkProtocol::deserialize_uint64(current_pos);
        current_pos += sizeof(uint64_t);

        hdr.version = NetworkProtocol::deserialize_uint8(current_pos);
        current_pos += sizeof(uint8_t);

        hdr.length = NetworkProtocol::deserialize_uint64(current_pos);
        // current_pos now conceptually points to the start of the payload.

        // Koan: The Version's Whisper - Is this a language we understand?
        if (hdr.version != NetworkProtocol::CURRENT_VERSION) {
            std::string err_msg =
                    "Unsupported protocol version. Expected: " + std::to_string(NetworkProtocol::CURRENT_VERSION) +
                    ", Got: " + std::to_string(hdr.version) + ".";
            // Even if the version is wrong, we might try to consume the message based on its stated length
            // to allow the stream to advance. The caller can then decide to discard it.
            // The 'INVALID_DATA' status will signal the issue.
            // Let's determine the full message size to advance past it.
            const uint64_t total_message_size_for_skip = NetworkProtocol::Header::SIZE + hdr.length;
            if (available_data_size < total_message_size_for_skip) {
                // Not enough data even to skip this invalid message based on its claimed length.
                return {std::nullopt, begin, ParseResultStatus::NEED_MORE_DATA,
                        err_msg + " Insufficient data to skip."};
            }
            // We can construct a "message" shell to return, but flag as invalid.
            NetworkProtocol::Message invalid_msg_shell;
            invalid_msg_shell.header = hdr;
            // Payload might be garbage or uninterpretable, but we can copy what's there.
            invalid_msg_shell.payload.assign(begin + NetworkProtocol::Header::SIZE,
                                             begin + total_message_size_for_skip);

            return {invalid_msg_shell, begin + total_message_size_for_skip, ParseResultStatus::INVALID_DATA, err_msg};
        }

        // Koan: The Payload's Promise - Do we have the data that was declared?
        const uint64_t total_message_size = NetworkProtocol::Header::SIZE + hdr.length;
        if (available_data_size < total_message_size) {
            return {std::nullopt, begin, ParseResultStatus::NEED_MORE_DATA, "Insufficient data for promised payload."};
        }

        // Koan: The Treasure Unveiled - Assembling the Full Message
        // All checks passed, and all data is present. Let's construct the message.
        NetworkProtocol::Message complete_message;
        complete_message.header = hdr;

        // The payload starts immediately after the header.
        auto payload_begin_iter = begin + NetworkProtocol::Header::SIZE;
        // The payload ends after 'hdr.length' bytes.
        auto payload_end_iter = payload_begin_iter + hdr.length;

        if (hdr.length > 0) { // Only assign if there's a payload to avoid issues with empty ranges for some vector ops
            complete_message.payload.assign(payload_begin_iter, payload_end_iter);
        }


        // Koan: The Next Step - Pointing to Where the Unparsed Data Begins
        // We've successfully consumed one full message.
        auto next_message_start_iter = begin + total_message_size;

        return {complete_message, next_message_start_iter, ParseResultStatus::SUCCESS, ""};
    }

} // namespace Parser
