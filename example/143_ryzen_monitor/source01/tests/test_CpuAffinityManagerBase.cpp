#include "CpuAffinityManagerBase.h"
#include <gtest/gtest.h>
#include <unistd.h>
TEST(CpuAffinityManagerBase, GetSelectedCpus_Initialized_ZeroBitset) {
  auto manager{CpuAffinityManagerBase(getpid())};
  auto expected_result{std::bitset<12>()};
  auto actual_result{manager.GetSelectedCpus()};
  EXPECT_EQ(actual_result, expected_result);
};