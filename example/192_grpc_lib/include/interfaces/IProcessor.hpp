#pragma once
#include <string>
#include <functional>
#include <memory>
#include "common/DataStructs.hpp"

class IProcessor {
public:
    virtual ~IProcessor() = default;

    // A callback provided by Alpha to generate a measurement from a timestamp
    using MeasurementProvider = std::function<Measurement(const Timestamp&)>;
    
    // A callback provided by Alpha to receive results
    using ResultReceiver = std::function<void(const Result&)>;

    // A callback provided by Alpha to receive the final summary
    using SummaryReceiver = std::function<void(const std::string&)>;

    virtual void start_session(
        MeasurementProvider measure_prov,
        ResultReceiver result_recv,
        SummaryReceiver summary_recv
    ) = 0;

    virtual void log(const std::string& message) = 0;
};

// Factory function signature for dynamic loading
extern "C" {
    IProcessor* create_processor();
}
