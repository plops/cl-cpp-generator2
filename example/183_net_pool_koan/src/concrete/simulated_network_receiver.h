//
// Created by martin on 5/13/25.
//

#ifndef SIMULATED_NETWORK_RECEIVER_H
#define SIMULATED_NETWORK_RECEIVER_H

#include "src/interfaces/inetwork_receiver.h"
#include "src/common/common.h"
#include <random>
#include <chrono>
#include <thread>
#include <cstring>
#include <arpa/inet.h> // htons
#include <atomic>
#include <iostream>

class SimulatedNetworkReceiver : public INetworkReceiver {
public:
    SimulatedNetworkReceiver(unsigned int seed = std::random_device{}())
        : rng_(seed), stop_flag_(false) {}

    std::optional<std::vector<std::byte>> receive_packet() override {
        if (stop_flag_.load(std::memory_order_acquire)) { // Acquire for visibility
            return std::nullopt;
        }
        // Simulate some delay
        // std::this_thread::sleep_for(std::chrono::milliseconds(1 + rng_() % 10));

        std::uniform_int_distribution<> type_dist(0, 2);
        PacketType type = static_cast<PacketType>(type_dist(rng_));
        std::vector<std::byte> packet;
        packet.push_back(static_cast<std::byte>(type));

        switch (type) {
            case PacketType::Image: {
                uint16_t len_net = htons(static_cast<uint16_t>(IMAGE_SIZE_BYTES));
                packet.resize(1 + sizeof(len_net) + IMAGE_SIZE_BYTES);
                std::memcpy(packet.data() + 1, &len_net, sizeof(len_net));
                for (size_t i = 0; i < IMAGE_SIZE_BYTES; ++i) {
                    packet[1 + sizeof(len_net) + i] = static_cast<std::byte>(rng_() % 256);
                }
                break;
            }
            case PacketType::Metadata: {
                Metadata meta = {static_cast<int>(rng_() % 1000), static_cast<float>(rng_()) / rng_.max()};
                packet.resize(1 + sizeof(Metadata));
                std::memcpy(packet.data() + 1, &meta, sizeof(Metadata));
                break;
            }
            case PacketType::Measurement: {
                Measurement meas = {static_cast<double>(rng_()) / rng_.max(), static_cast<double>(rng_()) / rng_.max() * 100.0};
                packet.resize(1 + sizeof(Measurement));
                std::memcpy(packet.data() + 1, &meas, sizeof(Measurement));
                break;
            }
            default: packet.clear(); break;
        }
        return packet;
    }
    void stop() override { stop_flag_.store(true, std::memory_order_release); } // Release for visibility
private:
    std::mt19937 rng_;
    std::atomic<bool> stop_flag_;
};
#endif //SIMULATED_NETWORK_RECEIVER_H
