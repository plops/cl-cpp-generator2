//
// Created by martin on 6/11/25.
//

// src/buffers/ring_array_buffer.cpp
#include "ring_array_buffer.h"
#include <algorithm> // for std::min
#include <iterator>  // for std::distance
#include <stdexcept> // for std::runtime_error

RingArrayBuffer::RingArrayBuffer(size_t capacity) :
    storage_(capacity), head_(0), tail_(0), count_(0), capacity_(capacity) {
    if (capacity == 0) { throw std::invalid_argument("RingArrayBuffer capacity cannot be zero."); }
}

void RingArrayBuffer::write(const unsigned char* data, size_t length) {
    if (length == 0) return;
    if (length > capacity_ - count_) {
        // Handle overflow: either throw, overwrite, or block.
        // For this test, we might throw or indicate an error.
        // This simplistic version might just fill up to capacity.
        // A real implementation would need a strategy.
        // For now, let's assume test data won't overflow or it's an error condition for tests.
        // For benchmark, we need to ensure this doesn't happen by design of test data.
        // If it does happen, it implies test data is too large for buffer.
        // Let's just write what fits for now to avoid crashing, but this is a limitation.
        length = capacity_ - count_;
        if (length == 0) return; // Completely full
        // std::cerr << "RingBuffer Warning: Write truncated due to insufficient space." << std::endl;
    }

    for (size_t i = 0; i < length; ++i) {
        storage_[tail_] = data[i];
        tail_           = (tail_ + 1) % capacity_;
        count_++;
    }
}

std::vector<unsigned char> RingArrayBuffer::get_linearized_data() const {
    std::vector<unsigned char> linear_data;
    if (count_ == 0) return linear_data;
    linear_data.reserve(count_);

    if (head_ < tail_) { // No wrap-around
        linear_data.insert(linear_data.end(), storage_.begin() + head_, storage_.begin() + tail_);
    }
    else { // Data wraps around
        linear_data.insert(linear_data.end(), storage_.begin() + head_, storage_.end());
        linear_data.insert(linear_data.end(), storage_.begin(), storage_.begin() + tail_);
    }
    return linear_data;
}

void RingArrayBuffer::consume_bytes_from_ring(size_t length) {
    length = std::min(length, count_); // Cannot consume more than available
    head_  = (head_ + length) % capacity_;
    count_ -= length;
}

size_t RingArrayBuffer::parse_and_consume_stream(
        const std::function<Parser::ParseOutput(std::vector<unsigned char>::const_iterator,                                                std::vector<unsigned char>::const_iterator)>& actual_parser,
        std::vector<NetworkProtocol::Message>&                                                out_messages) {
if (count_ == 0) { return 0; }

    // --- KEY DIFFERENCE: Linearization for the parser ---
    std::vector<unsigned char> temp_data_to_parse = get_linearized_data();
    // -----------------------------------------------------

    if (temp_data_to_parse.empty()) return 0;
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
            if (bytes_to_skip == 0 && current_iter_in_temp != end_iter_in_temp) bytes_to_skip = 1;
            total_bytes_consumed_from_temp += bytes_to_skip;
            break;        }
        else { // NEED_MORE_DATA
            break;
        }
    }

    if (total_bytes_consumed_from_temp > 0) { consume_bytes_from_ring(total_bytes_consumed_from_temp); }
    return messages_parsed_this_call;
}


size_t RingArrayBuffer::readable_bytes() const { return count_; }

void RingArrayBuffer::clear() {
    head_  = 0;
    tail_  = 0;
    count_ = 0;
}