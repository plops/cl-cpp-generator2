#include "LocalProcessor.hpp"
#include "../beta/SensorFusion.hpp"
#include <chrono>
#include <iostream>
#include <vector>

LocalProcessor::LocalProcessor() {}

void LocalProcessor::start_session(
    MeasurementProvider measure_prov,
    ResultReceiver result_recv,
    SummaryReceiver summary_recv
) {
    std::cout << "LocalProcessor: Starting local session." << std::endl;
    std::vector<Result> results;
    const int num_iterations = 10;

    for (int i = 0; i < num_iterations; ++i) {
        // 1. Generate Timestamp
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
        auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() % 1000000000;

        Timestamp ts{static_cast<uint64_t>(seconds), static_cast<uint32_t>(nanos)};

        // 2. Get measurement from Alpha
        Measurement m = measure_prov(ts);
        m.timestamp_id = i; // Ensure ID consistency

        // 3. Compute Result locally
        Result r = SensorFusion::fuse(m, ts);
        results.push_back(r);

        // 4. Send Result back to Alpha
        result_recv(r);
    }

    // Final summary
    summary_recv(SensorFusion::generate_summary(results));
    std::cout << "LocalProcessor: Session ended." << std::endl;
}

void LocalProcessor::log(const std::string& message) {
    // Local log just goes to console
    std::cout << "[LOCAL LOG] " << message << std::endl;
}

extern "C" IProcessor* create_processor() {
    return new LocalProcessor();
}
