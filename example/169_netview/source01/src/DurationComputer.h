//
// Created by martin on 3/17/25.
//

#ifndef DURATIONCOMPUTER_H
#define DURATIONCOMPUTER_H
#include <avcpp/timestamp.h>


/**
 * @brief Insert timestamps, with the second one this class returns the duration between consecutive timestamps
 */
class DurationComputer {
public:
    DurationComputer() = default;
    /**
     * @brief Add timestamp and computes difference to previous timestamp
     *
     * @param timestamp Timestamp of the current packet
     *
     * @return Duration in seconds between current and last timestamp. NAN if no previous timestamp
     */
    [[nodiscard]] double insert(av::Timestamp timestamp);

private:
    av::Timestamp previous;
    bool          isInitialized = false;
};


#endif // DURATIONCOMPUTER_H
