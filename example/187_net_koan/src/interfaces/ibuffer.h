//
// Created by martin on 6/11/25.
//
// src/interfaces/ibuffer.h
#pragma once

#include "parser.h"   // For Parser::ParseOutput
#include "protocol.h" // For NetworkProtocol::Message

#include <functional>
#include <memory> // For std::unique_ptr
#include <string>
#include <vector>

class IBuffer {
public:
    virtual ~IBuffer() = default;

    // Adds raw byte data to the buffer.
    virtual void write(const unsigned char* data, size_t length) = 0;

    // Attempts to parse one or more complete messages from the buffer's current content.
    // - actual_parser: The parsing function (e.g., Parser::parse_packet).
    // - out_messages: Vector to store successfully parsed messages.
    // Returns: The number of messages successfully parsed and consumed in this call.
    virtual size_t parse_and_consume_stream(
            const std::function<Parser::ParseOutput(std::vector<unsigned char>::const_iterator,
                                                    std::vector<unsigned char>::const_iterator)>& actual_parser,
            std::vector<NetworkProtocol::Message>&                                                out_messages) = 0;

    // Returns the number of bytes currently available for reading/parsing.
    virtual size_t readable_bytes() const = 0;

    // Clears the buffer, resetting its state.
    virtual void clear() = 0;

    // Returns the name of the buffer implementation (for logging/identification).
    virtual std::string name() const = 0;

    // Factory method for benchmark/test convenience
    static std::unique_ptr<IBuffer> create(const std::string& type, size_t fixed_capacity = 0);
};
