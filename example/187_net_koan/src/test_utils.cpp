//
// Created by martin on 6/9/25.
//
// src/test_utils.cpp
#include "test_utils.h"
#include <cstdlib> // for rand
#include <numeric> // for std::iota if needed for varied data

namespace TestUtils {

    std::vector<unsigned char> create_serialized_message(uint64_t id, uint8_t version,
                                                         const std::vector<unsigned char>& payload) {
        NetworkProtocol::Message msg;
        msg.header.id      = id;
        msg.header.version = version;
        msg.header.length  = payload.size();
        msg.payload        = payload;
        return NetworkProtocol::serializeMessage(msg);
    }

    std::vector<unsigned char> create_serialized_message(uint64_t id, const std::vector<unsigned char>& payload) {
        return create_serialized_message(id, NetworkProtocol::CURRENT_VERSION, payload);
    }


    std::vector<unsigned char> generate_packet_chunk(int num_messages, size_t avg_payload_size, uint64_t start_id) {
        std::vector<unsigned char> chunk;
        for (int i = 0; i < num_messages; ++i) {
            // Slightly vary payload size to make it more realistic
            size_t current_payload_size = avg_payload_size;
            if (avg_payload_size > 10) {
                // Vary by +/- 10%, ensuring it's not negative
                int variation = static_cast<int>(avg_payload_size / 10.0);
                if (variation > 0) {
                    current_payload_size =
                            std::max(size_t(1), avg_payload_size + (rand() % (2 * variation + 1)) - variation);
                }
            }
            else if (avg_payload_size == 0) { current_payload_size = 0; }


            std::vector<unsigned char> payload(current_payload_size);
            // Fill payload with some data (e.g., sequence of bytes)
            for (size_t j = 0; j < current_payload_size; ++j) {
                payload[j] = static_cast<unsigned char>((j + i) % 256);
            }

            auto serialized_msg = create_serialized_message(start_id + i, payload);
            chunk.insert(chunk.end(), serialized_msg.begin(), serialized_msg.end());
        }
        return chunk;
    }

} // namespace TestUtils
