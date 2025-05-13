//
// Created by martin on 5/13/25.
//

#ifndef PRODUCER_H
#define PRODUCER_H

#include "src/common/common.h"
#include "src/interfaces/ipool_interfaces.h"
#include "src/interfaces/inetwork_receiver.h"
#include <vector>
#include <stop_token>
#include <iostream>
#include <cstring>
#include <arpa/inet.h> // ntohs
/**
 * @brief Task function executed by the producer thread.
 * @details Continuously receives packets using INetworkReceiver, parses them, acquires a slot
 *          from the appropriate pool via IPoolProducer, deserializes the packet payload into the slot,
 *          and submits the slot index for consumption. Stops when the stop_token is requested or
 *          a non-recoverable error (like failure to acquire a pool slot after stop requested) occurs.
 *
 * @param stoken The stop_token used to signal cancellation.
 * @param network_receiver Reference to the network receiver implementation.
 * @param image_pool_producer Reference to the producer interface for the image pool.
 * @param metadata_pool_producer Reference to the producer interface for the metadata pool.
 * @param measurement_pool_producer Reference to the producer interface for the measurement pool.
 */
inline void producer_task(
    std::stop_token stoken,
    INetworkReceiver& network_receiver,
    IPoolProducer<Image>& image_pool_producer,
    IPoolProducer<Metadata>& metadata_pool_producer,
    IPoolProducer<Measurement>& measurement_pool_producer)
{
    std::cout << "Producer thread started." << std::endl;
    bool acquired_slot_in_iteration = true; // To check if loop should break

    while (!stoken.stop_requested() && acquired_slot_in_iteration) {
        acquired_slot_in_iteration = false; // Reset for current iteration
        std::optional<std::vector<std::byte>> packet_data_opt = network_receiver.receive_packet();

        if (stoken.stop_requested()) break; // Check after potentially blocking receive
        if (!packet_data_opt) {
            std::cout << "Producer: Network receiver returned no data or stopped. Exiting." << std::endl;
            break;
        }

        std::vector<std::byte>& packet_data = *packet_data_opt;
        if (packet_data.empty()) {
            std::cerr << "Producer: Received empty packet." << std::endl;
            acquired_slot_in_iteration = true; // Continue loop
            continue;
        }

        PacketType type = parse_packet_id(packet_data);
        const std::byte* data_ptr = packet_data.data() + 1;
        std::size_t data_size = packet_data.size() - 1;
        std::optional<std::size_t> index_opt;

        try {
            switch (type) {
                case PacketType::Image: {
                    if (data_size < sizeof(uint16_t)) { /* error */ acquired_slot_in_iteration = true; continue; }
                    uint16_t length_net;
                    std::memcpy(&length_net, data_ptr, sizeof(length_net));
                    uint16_t image_data_len = ntohs(length_net);
                    data_ptr += sizeof(uint16_t);
                    data_size -= sizeof(uint16_t);
                    if (image_data_len != IMAGE_SIZE_BYTES || data_size < image_data_len) { /* error */ acquired_slot_in_iteration = true; continue; }

                    index_opt = image_pool_producer.acquire_free_index();
                    if (!index_opt) break; // Break switch, acquired_slot_in_iteration remains false
                    Image& target_image = image_pool_producer.get_item_for_write(*index_opt);
                    if(target_image.size() != IMAGE_SIZE_BYTES) target_image.resize(IMAGE_SIZE_BYTES); // Defensive
                    std::memcpy(target_image.data(), data_ptr, IMAGE_SIZE_BYTES);
                    image_pool_producer.submit_filled_index(*index_opt);
                    acquired_slot_in_iteration = true;
                    break;
                }
                case PacketType::Metadata: {
                    if (data_size < sizeof(Metadata)) { /* error */ acquired_slot_in_iteration = true; continue; }
                    index_opt = metadata_pool_producer.acquire_free_index();
                    if (!index_opt) break;
                    Metadata& target_meta = metadata_pool_producer.get_item_for_write(*index_opt);
                    std::memcpy(&target_meta, data_ptr, sizeof(Metadata));
                    metadata_pool_producer.submit_filled_index(*index_opt);
                    acquired_slot_in_iteration = true;
                    break;
                }
                case PacketType::Measurement: {
                     if (data_size < sizeof(Measurement)) { /* error */ acquired_slot_in_iteration = true; continue; }
                    index_opt = measurement_pool_producer.acquire_free_index();
                    if (!index_opt) break;
                    Measurement& target_meas = measurement_pool_producer.get_item_for_write(*index_opt);
                    std::memcpy(&target_meas, data_ptr, sizeof(Measurement));
                    measurement_pool_producer.submit_filled_index(*index_opt);
                    acquired_slot_in_iteration = true;
                    break;
                }
                case PacketType::Unknown:
                    std::cerr << "Producer: Received unknown packet type." << std::endl;
                    acquired_slot_in_iteration = true; // Continue processing other packets
                    break;
            }
        } catch (const std::exception& e) {
            std::cerr << "Producer Error: Exception: " << e.what() << std::endl;
            // acquired_slot_in_iteration remains false, loop will exit if pool error
        }
         if (stoken.stop_requested()) break; // Check again before looping
    }
    std::cout << "Producer thread stopping." << std::endl;
}
#endif //PRODUCER_H
