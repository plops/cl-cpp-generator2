#include "CpuAffinityManager.h"
#include <gtest/gtest.h>
#include <unistd.h>
TEST CpuAffinityManager(GetSelectedCpus_Initialized_ZeroBitset) {
  auto manager{CpuAffinityManager(getpid())};
  auto expected_result{std::bitset<12>()};
  auto actual_result{manager.GetSelectedCpus()};
  EXPECT_EQ(actual_result, expected_result);
};