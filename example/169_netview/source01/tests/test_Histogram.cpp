//
// Created by martin on 3/22/25.
//

#include "Histogram.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <unistd.h>
TEST(Histogram_flaot, InsertData_InsertDataToEmpty_HaveOnePoint) {
    // Arrange

    // auto values{std::vector<float>({10.F, 11.F, 12.F})};
    auto histogram{Histogram<float, 3>(10.F, 12.F)};
    // Act

    histogram.insert(11.0F);
    // Assert

    EXPECT_EQ(histogram.getObservedMin(), 11.F);
    EXPECT_EQ(histogram.getObservedMax(), 11.F);
    EXPECT_EQ(histogram.getBinMin(), 10.F);
    EXPECT_EQ(histogram.getBinMax(), 12.F);
    EXPECT_EQ(histogram.getBinX(0), 10.F);
    EXPECT_EQ(histogram.getBinY(2), 12.F);
};