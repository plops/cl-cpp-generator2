//
// Created by martin on 3/22/25.
//

#include "Histogram.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <unistd.h>

class HistogramDoubleBaseTest : public ::testing::Test {
public:
  HistogramDoubleBaseTest() : histogram{10.,12.} {}

protected:
  void SetUp() final {
    // not needed
  }
  void TearDown() final {
    // not needed
  }
  Histogram<double, 3> histogram;
};

TEST_F(HistogramDoubleBaseTest, InsertData_InsertDataToEmpty_HaveBinExtrema) {
    histogram.insert(11.0);

    EXPECT_EQ(histogram.getBinMin(), 10.);
    EXPECT_EQ(histogram.getBinMax(), 12.);
};
TEST_F(HistogramDoubleBaseTest, InsertData_InsertDataToEmpty_HaveObservedExtrema) {
    histogram.insert(11.0F);

    EXPECT_EQ(histogram.getObservedMin(), 11.);
    EXPECT_EQ(histogram.getObservedMax(), 11.);
};