//
// Created by martin on 6/11/25.
//

// src/buffers/deque_buffer.cpp
#include "deque_buffer.h"
#include <iterator> // For std::distance
#include <vector>   // For temporary linearization

DequeBuffer::DequeBuffer() {}

void DequeBuffer::write(const unsigned char* data, size_t length) {
    buffer_.insert(buffer_.end(), data, data + length);
}

size_t DequeBuffer::parse_and_consume_stream(
        const std::function<Parser::ParseOutput(std::vector<unsigned char>::const_iterator,                                                std::vector<unsigned char>::const_iterator)>& actual_parser,
        std::vector<NetworkProtocol::Message>&                                                out_messages) {
if (buffer_.empty()) { return 0; }

    // --- KEY DIFFERENCE: Linearization for the parser ---
    // Create a temporary std::vector from the deque's content for the parser.
    // This copy is intentional to fit the existing parser and measure its cost.
    std::vector<unsigned char> temp_data_to_parse(buffer_.cbegin(), buffer_.cend());
    // -----------------------------------------------------

    if (temp_data_to_parse.empty()) return 0; // Should not happen if buffer_ wasn't empty

    auto   current_iter_in_temp           = temp_data_to_parse.cbegin();
    auto   end_iter_in_temp               = temp_data_to_parse.cend();
    size_t messages_parsed_this_call      = 0;
    size_t total_bytes_consumed_from_temp = 0;

    while (current_iter_in_temp != end_iter_in_temp) {
        Parser::ParseOutput result = actual_parser(current_iter_in_temp, end_iter_in_temp);

        if (result.status == Parser::ParseResultStatus::SUCCESS) {
            out_messages.push_back(result.message.value());
            messages_parsed_this_call++;
            size_t bytes_this_message = std::distance(current_iter_in_temp, result.next_data_iterator);
            total_bytes_consumed_from_temp += bytes_this_message;
            current_iter_in_temp = result.next_data_iterator;
        }
        else if (result.status == Parser::ParseResultStatus::INVALID_DATA) {
            size_t bytes_to_skip = std::distance(current_iter_in_temp, result.next_data_iterator);
            if (bytes_to_skip == 0 && current_iter_in_temp != end_iter_in_temp) bytes_to_skip = 1;            total_bytes_consumed_from_temp += bytes_to_skip;
            // current_iter_in_temp = result.next_data_iterator; // This was in temp vec
            break;        }
        else { // NEED_MORE_DATA
            break;
        }
    }

    if (total_bytes_consumed_from_temp > 0) {
        // Consume from the actual deque
        buffer_.erase(buffer_.cbegin(), buffer_.cbegin() + total_bytes_consumed_from_temp);
    }
    return messages_parsed_this_call;
}

size_t DequeBuffer::readable_bytes() const { return buffer_.size(); }

void DequeBuffer::clear() { buffer_.clear();
}