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

class HistogramIntBaseTest : public ::testing::Test {
public:
    HistogramIntBaseTest() : histogram{10,12} {}

protected:
    void SetUp() final {
        // not needed
    }
    void TearDown() final {
        // not needed
    }
    Histogram<int, 3> histogram;
};

TEST_F(HistogramDoubleBaseTest, InsertData_InsertDataToEmpty_HaveBinExtrema) {
    histogram.insert(11.0);

    EXPECT_EQ(histogram.getBinMin(), 10.);
    EXPECT_EQ(histogram.getBinMax(), 12.);
};
TEST_F(HistogramDoubleBaseTest, InsertData_InsertDataToEmpty_HaveObservedExtrema) {
    histogram.insert(11.0);
    EXPECT_EQ(histogram.getBinMin(), 10.);
    EXPECT_EQ(histogram.getBinMax(), 12.);
    EXPECT_EQ(histogram.getObservedMin(), 11.);
    EXPECT_EQ(histogram.getObservedMax(), 11.);
};
TEST_F(HistogramDoubleBaseTest, InsertData_Insert2DataToEmpty_HaveObservedExtrema) {
    histogram.insert(11.0);
    histogram.insert(12.0);
    EXPECT_EQ(histogram.getBinMin(), 10.);
    EXPECT_EQ(histogram.getBinMax(), 12.);
    EXPECT_EQ(histogram.getObservedMin(), 11.);
    EXPECT_EQ(histogram.getObservedMax(), 12.);
};
TEST_F(HistogramDoubleBaseTest, InsertData_Insert3OutOfRangeDataToEmpty_Histogram) {
    histogram.insert(9.0);
    histogram.insert(11.0);
    histogram.insert(13.0);
    EXPECT_EQ(histogram.getBinMin(), 10.);
    EXPECT_EQ(histogram.getBinMax(), 12.);
    EXPECT_EQ(histogram.getObservedMin(), 9.);
    EXPECT_EQ(histogram.getObservedMax(), 13.);
    EXPECT_EQ(histogram.getBinY(0),1);
    EXPECT_EQ(histogram.getBinY(1),1);
    EXPECT_EQ(histogram.getBinY(2),1);
};
TEST_F(HistogramDoubleBaseTest, InsertData_Insert3DataToEmpty_HaveObservedExtremaAndHistogram) {
    histogram.insert(10.0);
    histogram.insert(11.0);
    histogram.insert(12.0);
    EXPECT_EQ(histogram.getBinMin(), 10.);
    EXPECT_EQ(histogram.getBinMax(), 12.);
    EXPECT_EQ(histogram.getObservedMin(), 10.);
    EXPECT_EQ(histogram.getObservedMax(), 12.);
    EXPECT_EQ(histogram.getBinY(0),1);
    EXPECT_EQ(histogram.getBinY(1),1);
    EXPECT_EQ(histogram.getBinY(2),1);
};
TEST_F(HistogramIntBaseTest, InsertData_Insert3OutOfRangeDataToEmpty_HaveObservedExtremaAndHistogram) {
    histogram.insert(9);
    histogram.insert(11);
    histogram.insert(13);
    EXPECT_EQ(histogram.getBinMin(), 10);
    EXPECT_EQ(histogram.getBinMax(), 12);
    EXPECT_EQ(histogram.getObservedMin(), 9);
    EXPECT_EQ(histogram.getObservedMax(), 13);
    EXPECT_EQ(histogram.getElementCount(), 3);
    EXPECT_EQ(histogram.getBinY(0),1);
    EXPECT_EQ(histogram.getBinY(1),1);
    EXPECT_EQ(histogram.getBinY(2),1);
};

TEST_F(HistogramIntBaseTest, InsertData_Insert4OutOfRangeDataToEmpty_HaveObservedExtremaAndHistogram) {
    histogram.insert(-1);
    histogram.insert(9);
    histogram.insert(11);
    histogram.insert(13);
    EXPECT_EQ(histogram.getBinMin(), 10);
    EXPECT_EQ(histogram.getBinMax(), 12);
    EXPECT_EQ(histogram.getObservedMin(), -1);
    EXPECT_EQ(histogram.getObservedMax(), 13);
    EXPECT_EQ(histogram.getElementCount(), 4);
    EXPECT_EQ(histogram.getBinY(0),2);
    EXPECT_EQ(histogram.getBinY(1),1);
    EXPECT_EQ(histogram.getBinY(2),1);
};
