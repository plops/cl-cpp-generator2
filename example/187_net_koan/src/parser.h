//
// Created by martin on 6/8/25.
//

// src/parser.h
#pragma once

#include "protocol.h"
#include <vector>
#include <optional> // C++17 for optional return
#include <iterator> // For std::distance

namespace Parser {

    // Koan: The Parser's Verdict - What was Found in the Stream
    enum class ParseResultStatus {
        SUCCESS,         // A complete message was parsed.
        NEED_MORE_DATA,  // The data is valid so far, but incomplete for a full message.
        INVALID_DATA     // The data is malformed or violates protocol rules (e.g., bad version).
    };

    // Koan: The Parser's Report - The Message and the Way Forward
    // This structure tells us what message (if any) was parsed,
    // the status of the parsing attempt, and where to look for the next message.
    struct ParseOutput {
        std::optional<NetworkProtocol::Message> message;             // The parsed message, if successful.
        std::vector<unsigned char>::const_iterator next_data_iterator; // Iterator to the start of the *next* piece of data.
        // If SUCCESS: after the parsed message.
        // If NEED_MORE_DATA: same as input 'begin'.
        // If INVALID_DATA: attempts to point past the invalid segment if possible.
        ParseResultStatus status;
        std::string error_message; // Provides context for errors or incomplete data.
    };

    // Koan: The Act of Parsing - The Gatekeeper Function
    // Takes a window into a byte buffer (via iterators) and attempts to extract one message.
    ParseOutput parse_packet(std::vector<unsigned char>::const_iterator begin,
                             std::vector<unsigned char>::const_iterator end);

} // namespace Parser
