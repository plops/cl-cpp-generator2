//
// Created by martin on 6/11/25.
//

// src/buffers/ring_array_buffer.h
#pragma once
#include <array>
#include <vector> // For linearization
#include "interfaces/ibuffer.h"

// Define a sensible default capacity, can be overridden in constructor
constexpr size_t DEFAULT_RING_CAPACITY = 16 * 1024; // 16KB

class RingArrayBuffer : public IBuffer {
public:
    explicit RingArrayBuffer(size_t capacity = DEFAULT_RING_CAPACITY);
    void   write(const unsigned char* data, size_t length) override;
    size_t parse_and_consume_stream(
            const std::function<Parser::ParseOutput(std::vector<unsigned char>::const_iterator,
                                                    std::vector<unsigned char>::const_iterator)>& actual_parser,
            std::vector<NetworkProtocol::Message>& out_messages) override;
    size_t      readable_bytes() const override;
    void        clear() override;
    std::string name() const override { return "RingArrayBuffer"; }

private:
    std::vector<unsigned char> storage_; // Using vector for dynamic capacity setting, acts like array
    size_t                     head_;    // Read pointer (index)
    size_t                     tail_;    // Write pointer (index)
    size_t                     count_;   // Number of bytes currently in buffer
    const size_t               capacity_;

    // Helper to linearize data for the parser
    std::vector<unsigned char> get_linearized_data() const;
    void                       consume_bytes_from_ring(size_t length);
};
