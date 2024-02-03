#include "CpuAffinityManagerBase.h"
#include <gtest/gtest.h>
#include <thread>
#include <unistd.h>
TEST(CpuAffinityManagerBase, GetSelectedCpus_Initialized_FullBitset) {
  auto n{std::thread::hardware_concurrency()};
  auto manager{CpuAffinityManagerBase(getpid(), n)};
  // FIXME: this only works on a twelve core cpu

  auto expected_result{std::vector<bool>(n, true)};
  auto actual_result{manager.GetSelectedCpus()};
  EXPECT_EQ(actual_result, expected_result);
};