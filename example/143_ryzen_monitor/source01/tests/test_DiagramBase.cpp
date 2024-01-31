#include "DiagramBase.h"
#include <gtest/gtest.h>
#include <unistd.h>
TEST(DiagramBase, AddDataPoint_AddPointToEmpty_HaveOnePoint) {
  // Arrange

  auto diagram{DiagramBase({{1.0F, 0.F, 0.F, 1.0F}}, 12, 10)};
  auto expected_result{1};
  // Act

  diagram.AddDataPoint(0, 1.0F, 10.F);
  EXPECT_EQ(diagram.GetTimePoints().size(), 1);
  EXPECT_EQ(diagram.GetDiagrams()[0].values.size(), 1);
};