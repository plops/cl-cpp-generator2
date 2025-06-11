//
// Created by martin on 6/11/25.
//
// src/buffers/vector_buffer.h
#pragma once
#include <vector>
#include "interfaces/ibuffer.h"

class VectorBuffer : public IBuffer {
public:
    VectorBuffer();
    void   write(const unsigned char* data, size_t length) override;
    size_t parse_and_consume_stream(
            const std::function<Parser::ParseOutput(std::vector<unsigned char>::const_iterator,
                                                    std::vector<unsigned char>::const_iterator)>& actual_parser,
            std::vector<NetworkProtocol::Message>& out_messages) override;
    size_t      readable_bytes() const override;
    void        clear() override;
    std::string name() const override { return "VectorBuffer"; }

private:
    std::vector<unsigned char> buffer_;
    size_t                     read_offset_;                  // Indicates start of unparsed data within buffer_
    const size_t               erase_threshold_bytes_ = 4096; // When to perform erase
    const double               erase_threshold_ratio_ = 0.5;  // When read_offset_ is this ratio of total size
};
