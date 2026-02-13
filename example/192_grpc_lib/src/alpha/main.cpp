#include <iostream>
#include <string>
#include <memory>
#include "interfaces/IProcessor.hpp"

int main() {
    std::cout << "Alpha: Initializing..." << std::endl;

    // create_processor is linked from either libone or libtwo
    std::unique_ptr<IProcessor> processor(create_processor());

    if (!processor) {
        std::cerr << "Alpha: Failed to create processor." << std::endl;
        return 1;
    }

    processor->log("Session starting from Alpha.");

    processor->start_session(
        [](const Timestamp& ts) -> Measurement {
            std::cout << "Alpha: Providing measurement for T=" << ts.seconds << "." << ts.nanos << std::endl;
            return Measurement{
                .value = 42.0 + (ts.seconds % 10),
                .timestamp_id = 0 // Will be set by processor
            };
        },
        [](const Result& res) {
            std::cout << "Alpha: Received Result=" << res.fused_value << " for ID=" << res.timestamp_id << std::endl;
        },
        [](const std::string& summary) {
            std::cout << "Alpha: Received Summary: " << summary << std::endl;
        }
    );

    processor->log("Session finished in Alpha.");
    std::cout << "Alpha: Execution complete." << std::endl;

    return 0;
}
