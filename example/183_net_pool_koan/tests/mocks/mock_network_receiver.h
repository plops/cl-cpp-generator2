//
// Created by martin on 5/13/25.
//

#ifndef MOCK_NETWORK_RECEIVER_H
#define MOCK_NETWORK_RECEIVER_H

#include "src/interfaces/inetwork_receiver.h"
#include <gmock/gmock.h>

class MockNetworkReceiver : public INetworkReceiver {
public:
    MOCK_METHOD(std::optional<std::vector<std::byte>>, receive_packet, (), (override));
    MOCK_METHOD(void, stop, (), (override));
};
#endif //MOCK_NETWORK_RECEIVER_H
