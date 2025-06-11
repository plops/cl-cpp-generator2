//
// Created by martin on 6/11/25.
//
// src/buffers/deque_buffer.h
#pragma once
#include <deque>
#include "interfaces/ibuffer.h"

class DequeBuffer : public IBuffer {
public:
    DequeBuffer();
    void   write(const unsigned char* data, size_t length) override;
    size_t parse_and_consume_stream(
            const std::function<Parser::ParseOutput(std::vector<unsigned char>::const_iterator,
                                                    std::vector<unsigned char>::const_iterator)>& actual_parser,
            std::vector<NetworkProtocol::Message>& out_messages) override;
    size_t      readable_bytes() const override;
    void        clear() override;
    std::string name() const override { return "DequeBuffer"; }

private:
    std::deque<unsigned char> buffer_;
};
