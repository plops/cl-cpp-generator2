//
// Created by martin on 5/13/25.
//

#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <cstddef> // For std::byte
#include <cstdint> // For uintX_t types
#include <string>  // For std::to_integer in C++20 for std::byte

// --- Constants ---
constexpr std::size_t IMAGE_WIDTH = 128;
constexpr std::size_t IMAGE_HEIGHT = 128;
constexpr std::size_t IMAGE_SIZE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT;

constexpr std::size_t IMAGE_POOL_SIZE = 10;
constexpr std::size_t METADATA_POOL_SIZE = 12;
constexpr std::size_t MEASUREMENT_POOL_SIZE = 20;

// --- Data Types ---

/**
 * @brief Type alias for image data.
 * @details Represents a fixed-size image stored as a vector of bytes.
 * The actual storage is managed within a DataPool.
 */
using Image = std::vector<std::byte>;

/**
 * @brief Structure holding metadata received from the network.
 */
struct Metadata {
    int i;
    float f;
    bool operator==(const Metadata& other) const { // For testing with Eq matcher
        return i == other.i && f == other.f;
    }
};

struct Measurement {
    double q;
    double p;
     bool operator==(const Measurement& other) const { // For testing
        return q == other.q && p == other.p;
    }
};

// --- Network Packet IDs ---

/**
 * @brief Enumerates the possible types of data packets received over the network.
 */
enum class PacketType : uint8_t {
    Image = 0,
    Metadata = 1,
    Measurement = 2,
    Unknown = 255
};

/**
 * @brief Parses the packet type ID from the raw packet data.
 * @param packet_data The raw bytes received from the network.
 * @return The PacketType enum value corresponding to the first byte, or PacketType::Unknown if invalid/empty.
 */
inline PacketType parse_packet_id(const std::vector<std::byte>& packet_data) {
    if (packet_data.empty()) {
        return PacketType::Unknown;
    }
    uint8_t id_val = static_cast<uint8_t>(packet_data[0]);
    switch (id_val) {
        case 0: return PacketType::Image;
        case 1: return PacketType::Metadata;
        case 2: return PacketType::Measurement;
        default: return PacketType::Unknown;
    }
}
#endif //COMMON_H
