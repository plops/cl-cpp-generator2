#include "SensorFusion.hpp"
#include <sstream>
#include <cmath>
#include <numeric>

Result SensorFusion::fuse(const Measurement& measurement, const Timestamp& server_time) {
    // A simple fusion logic: 
    // Measurement value + some server-side contribution derived from the current timestamp
    double server_contribution = std::sin(static_cast<double>(server_time.seconds % 360)) * 0.5;
    
    return Result{
        .fused_value = measurement.value + server_contribution,
        .timestamp_id = measurement.timestamp_id
    };
}

std::string SensorFusion::generate_summary(const std::vector<Result>& results) {
    if (results.empty()) return "No data processed.";

    double sum = std::accumulate(results.begin(), results.end(), 0.0, 
        [](double s, const Result& r) { return s + r.fused_value; });
    double avg = sum / results.size();

    std::ostringstream oss;
    oss << "Session Summary: Processed " << results.size() << " measurements. "
        << "Average value: " << avg;
    return oss.str();
}
