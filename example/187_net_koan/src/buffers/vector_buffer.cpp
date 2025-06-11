//
// Created by martin on 6/11/25.
//

// src/buffers/vector_buffer.cpp
#include "vector_buffer.h"
#include <algorithm> // for std::min
#include <iterator>  // for std::distance

VectorBuffer::VectorBuffer() : read_offset_(0) {}

void VectorBuffer::write(const unsigned char* data, size_t length) {
    // If there's processed data at the beginning that hasn't been erased,
    // and new data would cause a large allocation, consider erasing first.
    // For simplicity here, we always append then potentially erase in parse_and_consume.
    buffer_.insert(buffer_.end(), data, data + length);
}

size_t VectorBuffer::parse_and_consume_stream(
        const std::function<Parser::ParseOutput(std::vector<unsigned char>::const_iterator,
                                                std::vector<unsigned char>::const_iterator)>& actual_parser,
        std::vector<NetworkProtocol::Message>&                                                out_messages) {
    if (read_offset_ >= buffer_.size()) {
        // All existing data parsed or buffer empty from start.
        // If buffer is large and fully processed, good time to clear it.
        if (read_offset_ > 0 && read_offset_ == buffer_.size()) {
            buffer_.clear();
            read_offset_ = 0;
        }
        return 0;
    }

    auto   current_parse_iter                = buffer_.cbegin() + read_offset_;
    auto   end_of_buffer_iter                = buffer_.cend();
    size_t messages_parsed_this_call         = 0;
    size_t initial_read_offset_for_this_call = read_offset_;

    while (current_parse_iter != end_of_buffer_iter) {
        Parser::ParseOutput result = actual_parser(current_parse_iter, end_of_buffer_iter);

        if (result.status == Parser::ParseResultStatus::SUCCESS) {
            out_messages.push_back(result.message.value());
            messages_parsed_this_call++;
            current_parse_iter = result.next_data_iterator;
            // Update read_offset_ based on how many bytes from the *start* of buffer_ are now processed
            read_offset_ = std::distance(buffer_.cbegin(), current_parse_iter);
        }
        else if (result.status == Parser::ParseResultStatus::INVALID_DATA) {
            // Simplistic error handling: skip the problematic segment if parser advanced, otherwise stop.
            if (result.next_data_iterator > current_parse_iter) {
                current_parse_iter = result.next_data_iterator;
                read_offset_       = std::distance(buffer_.cbegin(), current_parse_iter);
            }
            else {
                // Parser didn't advance, to avoid infinite loop, advance by one byte if possible.
                // This is a basic recovery, real systems might need more sophisticated logic.
                if (current_parse_iter != end_of_buffer_iter) {
                    current_parse_iter++;
                    read_offset_ = std::distance(buffer_.cbegin(), current_parse_iter);
                }
            }
            // For benchmark/test, we might log this error but continue if possible or stop
            break;
        }
        else {     // NEED_MORE_DATA
            break; // Cannot parse further with current data
        }
    }

    // Heuristic for erasing processed data from the front of the vector
    if (read_offset_ > 0) {
        bool should_erase = (read_offset_ >= erase_threshold_bytes_);
        should_erase      = should_erase ||
                (buffer_.size() > 0 && (static_cast<double>(read_offset_) / buffer_.size()) >= erase_threshold_ratio_);

        if (read_offset_ == buffer_.size()) { // All data processed
            should_erase = true;
        }

        if (should_erase) {
            buffer_.erase(buffer_.begin(), buffer_.begin() + read_offset_);
            read_offset_ = 0;
        }
    }
    return messages_parsed_this_call;
}

size_t VectorBuffer::readable_bytes() const { return buffer_.size() - read_offset_; }

void VectorBuffer::clear() {
    buffer_.clear();
    read_offset_ = 0;
}
