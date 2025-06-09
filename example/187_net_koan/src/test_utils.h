//
// Created by martin on 6/9/25.
//

// src/test_utils.h
#pragma once

#include <string>
#include <vector>
#include "protocol.h"

namespace TestUtils {

    // Generates a serialized message
    std::vector<unsigned char> create_serialized_message(uint64_t id, uint8_t version,
                                                         const std::vector<unsigned char>& payload);

    std::vector<unsigned char> create_serialized_message(
            uint64_t                          id,
            const std::vector<unsigned char>& payload); // Uses CURRENT_VERSION

    // Generates a chunk of data containing multiple serialized messages
    std::vector<unsigned char> generate_packet_chunk(int num_messages, size_t avg_payload_size, uint64_t start_id = 0);

} // namespace TestUtils
