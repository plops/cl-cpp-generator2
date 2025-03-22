//
// Created by martin on 3/22/25.
//

#include "DurationComputer.h"

#include <gtest/gtest.h>
#include <array>
#include <vector>

using namespace std;

class DurationComputerBaseTest : public ::testing::Test {
public:
    DurationComputerBaseTest() = default;

protected:
    void SetUp() final {
        for (const auto& e: times) {
            const av::Rational base{1,1};
            const av::Timestamp ts{e,base};
            results.push_back(c.insert(ts));
        }
    }
    void TearDown() final {
        // not needed
    }
    DurationComputer c{};
    std::vector<double> results{};
    array<int,4> times{1,2,4,5};
};

TEST_F(DurationComputerBaseTest, Sequence_Compute_ResultCorrect) {
    ASSERT_EQ(results.size(), times.size());
    ASSERT_TRUE(isnan(results[0]));
    ASSERT_DOUBLE_EQ(results[1], 1.);
    ASSERT_DOUBLE_EQ(results[2], 2.);
    ASSERT_DOUBLE_EQ(results[3], 1.);
};