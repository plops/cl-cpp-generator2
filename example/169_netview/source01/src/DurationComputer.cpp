//
// Created by martin on 3/17/25.
//

#include "DurationComputer.h"
double DurationComputer::insert(av::Timestamp timestamp) {
    if (!isInitialized) {
        previous = timestamp;
        isInitialized = true;
        return std::nan("1");
    }
    auto duration = timestamp.seconds() - previous.seconds();
    previous = timestamp;
    return duration;
}
