//
// Created by martin on 5/13/25.
//

#ifndef CONSUMER_H
#define CONSUMER_H

#include "src/interfaces/ipool_interfaces.h"
#include "src/interfaces/iitem_processor.h"
#include "src/core/pool_item_reference.h" // For PoolItemReference type itself
#include <stop_token>
#include <iostream>
#include <string_view>
/**
 * @brief Template task function executed by consumer threads.
 * @details Continuously attempts to consume items from the data source (pool) using an RAII wrapper (PoolItemReference).
 *          If an item is successfully consumed, it delegates the processing of that item to the provided IItemProcessor.
 *          The PoolItemReference automatically returns the item's index to the pool when it goes out of scope.
 *          Stops when the stop_token is requested or the data source indicates no more items are available (e.g., pool stopped).
 *
 * @tparam T The type of data item being consumed and processed.
 * @param stoken The stop_token used to signal cancellation.
 * @param consumer_name A descriptive name for the consumer thread (used for logging).
 * @param data_source Reference to the pool consumer interface providing the items.
 * @param item_processor Reference to the processor responsible for handling consumed items.
 */
template <typename T>
inline void consumer_task(
    std::stop_token stoken,
    std::string_view consumer_name,
    IPoolConsumer<T>& data_source,
    IItemProcessor<T>& item_processor)
{
    std::cout << consumer_name << " thread started." << std::endl;
    while (!stoken.stop_requested()) {
        std::optional<PoolItemReference<T>> item_ref_opt = data_source.consume_item_ref();

        if (stoken.stop_requested() && !item_ref_opt) break; // Prefer stop_token check if item_ref is nullopt due to stop

        if (!item_ref_opt) {
            if (!stoken.stop_requested()){ // Only log if not already stopping
                 std::cout << consumer_name << ": Pool queue returned nullopt. Exiting." << std::endl;
            }
            break;
        }
        item_processor.process(item_ref_opt->get(), item_ref_opt->index());
        // RAII handles returning index via item_ref_opt destructor
    }
    std::cout << consumer_name << " thread stopping." << std::endl;
}
#endif //CONSUMER_H
