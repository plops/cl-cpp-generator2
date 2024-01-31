#include "DiagramBase.h"
#include <gtest/gtest.h>
#include <unistd.h>
TEST(DiagramBase, AddDataPoint_AddPointToEmpty_HaveOnePoint) {
  // Arrange

  auto values{std::vector<float>({10.F, 11.F})};
  auto diagram{DiagramBase({{1.0F, 0.F, 0.F, 1.0F}}, values.size(), 10)};
  // Act

  diagram.AddDataPoint(1.0F, values);
  // Assert

  EXPECT_EQ(diagram.GetTimePoints().size(), 1);
  EXPECT_EQ(diagram.GetDiagrams()[0].values.size(), 1);
};
TEST(DiagramBase, AddDataPoint_AddPointToOne_HaveTwoPoints) {
  // Arrange

  auto values{std::vector<float>({10.F, 11.F})};
  auto diagram{DiagramBase({{1.0F, 0.F, 0.F, 1.0F}}, values.size(), 10)};
  // Act

  diagram.AddDataPoint(1.0F, values);
  diagram.AddDataPoint(2.0F, values);
  // Assert

  EXPECT_EQ(diagram.GetTimePoints().size(), 2);
  EXPECT_EQ(diagram.GetDiagrams()[0].values.size(), 2);
};
TEST(DiagramBase, AddDataPoint_AddLastPoint_HaveThreePoints) {
  // Arrange

  auto values{std::vector<float>({10.F, 11.F})};
  auto diagram{DiagramBase({{1.0F, 0.F, 0.F, 1.0F}}, values.size(), 3)};
  // Act

  diagram.AddDataPoint(1.0F, values);
  diagram.AddDataPoint(2.0F, values);
  diagram.AddDataPoint(3.0F, values);
  // Assert

  EXPECT_EQ(diagram.GetTimePoints().size(), 3);
  EXPECT_EQ(diagram.GetDiagrams()[0].values.size(), 3);
};
TEST(DiagramBase, AddDataPoint_AddOneMorePointsThanFit_HaveThreePoints) {
  // Arrange

  auto values{std::vector<float>({10.F, 11.F})};
  auto diagram{DiagramBase({{1.0F, 0.F, 0.F, 1.0F}}, 2, 3)};
  // Act

  diagram.AddDataPoint(1.0F, {10.F, 1.00e+2F});
  diagram.AddDataPoint(2.0F, {20.F, 2.00e+2F});
  diagram.AddDataPoint(3.0F, {30.F, 3.00e+2F});
  diagram.AddDataPoint(4.0F, {40.F, 4.00e+2F});
  // Assert

  EXPECT_EQ(diagram.GetTimePoints().size(), 3);
  EXPECT_EQ(diagram.GetTimePoints().at(2), 4.0F);
  EXPECT_EQ(diagram.GetDiagrams().at(0).values.size(), 3);
  EXPECT_EQ(diagram.GetDiagrams().at(0).values.at(2), 40.F);
};