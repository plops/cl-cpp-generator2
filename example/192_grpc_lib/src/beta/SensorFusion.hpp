#pragma once
#include "common/DataStructs.hpp"
#include <string>
#include <vector>

class SensorFusion {
public:
    // Compute a result based on client measurement and server state
    static Result fuse(const Measurement& measurement, const Timestamp& server_time);

    // Final summary computation
    static std::string generate_summary(const std::vector<Result>& results);
};
