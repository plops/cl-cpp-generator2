//
// Created by martin on 5/13/25.
//

#ifndef INETWORK_RECEIVER_H
#define INETWORK_RECEIVER_H

#include <vector>
#include <cstddef> // std::byte
#include <optional>
/**
 * @brief Interface for components responsible for receiving raw data packets, typically from a network socket.
 */
class INetworkReceiver {
public:
    virtual ~INetworkReceiver() = default;
    /**
     * @brief Attempts to receive a single data packet.
     * @details This method may block until a packet is available or a stop condition occurs.
     * @return An std::optional containing the raw bytes of the received packet,
     *         or std::nullopt if the receiver is stopped or an error occurred preventing reception.
     */
    virtual std::optional<std::vector<std::byte>> receive_packet() = 0;
    /**
     * @brief Signals the receiver to stop its operation.
     * @details This should cause any blocking `receive_packet` calls to eventually return (typically with std::nullopt).
     */
    virtual void stop() = 0;
};
#endif //INETWORK_RECEIVER_H
