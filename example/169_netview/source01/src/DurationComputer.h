//
// Created by martin on 3/17/25.
//

#ifndef DURATIONCOMPUTER_H
#define DURATIONCOMPUTER_H
#include <timestamp.h>


/**
 * @brief Insert timestamps, with the second one this class returns the duration between consecutive timestamps
 */
class DurationComputer {
    public:
    DurationComputer() = default;
    [[nodiscard]] double insert(av::Timestamp timestamp);
private:
    av::Timestamp previous;
    bool isInitialized = false;
};



#endif //DURATIONCOMPUTER_H
