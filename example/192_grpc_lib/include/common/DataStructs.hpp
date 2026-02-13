#pragma once
#include <cstdint>

struct Timestamp {
    uint64_t seconds;
    uint32_t nanos;
};

struct Measurement {
    double value;
    uint64_t timestamp_id;
};

struct Result {
    double fused_value;
    uint64_t timestamp_id;
};
