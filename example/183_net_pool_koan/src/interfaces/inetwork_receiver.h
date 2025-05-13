//
// Created by martin on 5/13/25.
//

#ifndef INETWORK_RECEIVER_H
#define INETWORK_RECEIVER_H

#include <vector>
#include <cstddef> // std::byte
#include <optional>

class INetworkReceiver {
public:
    virtual ~INetworkReceiver() = default;
    virtual std::optional<std::vector<std::byte>> receive_packet() = 0;
    virtual void stop() = 0;
};
#endif //INETWORK_RECEIVER_H
