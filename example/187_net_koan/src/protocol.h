//
// Created by martin on 6/8/25.
//

// src/protocol.h
#pragma once

#include <arpa/inet.h> // For ntohl, htonl, htobe64, be64toh (ensure your system provides these for 64-bit)
#include <cstdint>
#include <cstring>   // For std::memcpy
#include <sstream>   // For Message::toString
#include <stdexcept> // For std::runtime_error
#include <string>    // For Message::toString
#include <vector>

// If htobe64/be64toh are not available on an older system, you might need manual conversion
// or ensure your compiler/libc provides them (often via _BSD_SOURCE or _GNU_SOURCE macros,
// but modern POSIX should have them). For this Koan, we assume they exist.

namespace NetworkProtocol {

    const uint8_t CURRENT_VERSION = 0x01;

    // Koan: The Header - The Face of a Message
    // Every message introduces itself with this structure.
    struct Header {
        uint64_t id;      // A unique (or sequential) identifier for the message
        uint8_t  version; // The version of this protocol
        uint64_t length;  // The length of the payload *following* this header

        static constexpr size_t SIZE = sizeof(id) + sizeof(version) + sizeof(length);

        // A helper to describe the header
        std::string toString() const {
            std::stringstream ss;
            ss << "ID: " << id << ", Version: " << static_cast<int>(version) << ", PayloadLength: " << length;
            return ss.str();
        }
    };

    // Koan: The Message - Header and Essence Combined
    // A fully formed message, holding its credentials and its content.
    struct Message {
        Header                     header;
        std::vector<unsigned char> payload;

        std::string toString() const {
            std::stringstream ss;
            ss << "Message { " << header.toString() << ", PayloadSize: " << payload.size();
            // Optionally, print some payload data if it's text, be careful with binary data
            // if (payload.size() < 50) { // Print small payloads
            //    ss << ", PayloadHex: ";
            //    for(size_t i = 0; i < payload.size(); ++i) {
            //        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(payload[i]);
            //    }
            // }
            ss << " }";
            return ss.str();
        }
    };

    // Koan: The Art of Transmutation - Serialization Helpers
    // We must translate our numbers into a stream of bytes for the network.
    // And from a stream of bytes back into numbers upon reception.
    // These helpers ensure numbers are in Network Byte Order (Big Endian).

    inline void serialize_uint64(uint64_t val, unsigned char* buffer) {
        uint64_t net_val = htobe64(val); // Host To Big-Endian 64-bit
        std::memcpy(buffer, &net_val, sizeof(net_val));
    }

    inline void serialize_uint8(uint8_t val, unsigned char* buffer) {
        *buffer = val; // Single byte, no endianness issue
    }

    inline uint64_t deserialize_uint64(const unsigned char* buffer) {
        uint64_t net_val;
        std::memcpy(&net_val, buffer, sizeof(net_val));
        return be64toh(net_val); // Big-Endian To Host 64-bit
    }

    inline uint8_t deserialize_uint8(const unsigned char* buffer) { return *buffer; }

    // Koan: Crafting the Message for Travel - Serializing a Full Message
    // Takes a Message object and turns it into a sequence of bytes ready for sending.
    inline std::vector<unsigned char> serializeMessage(const Message& msg) {
        if (msg.payload.size() != msg.header.length) {
            throw std::runtime_error("Message integrity error: header.length does not match payload.size()");
        }
        std::vector<unsigned char> buffer(Header::SIZE + msg.header.length);
        unsigned char*             ptr = buffer.data();

        serialize_uint64(msg.header.id, ptr);
        ptr += sizeof(uint64_t);

        serialize_uint8(msg.header.version, ptr);
        ptr += sizeof(uint8_t);

        serialize_uint64(msg.header.length, ptr);
        ptr += sizeof(uint64_t);

        if (!msg.payload.empty()) { std::memcpy(ptr, msg.payload.data(), msg.header.length); }
        return buffer;
    }

} // namespace NetworkProtocol
