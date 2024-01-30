#include "CpuAffinityManagerBase.h"
#include <gtest/gtest.h>
#include <unistd.h>
TEST(CpuAffinityManagerBase, GetSelectedCpus_Initialized_FullBitset) {
  auto manager{CpuAffinityManagerBase(getpid())};
  // FIXME: this only works on a twelf core cpu

  auto expected_result{std::bitset<12>("111111111111")};
  auto actual_result{manager.GetSelectedCpus()};
  EXPECT_EQ(actual_result, expected_result);
};
TEST(CpuAffinityManagerBase, SetSelectedCpus_Set_ValidBitset) {
  auto manager{CpuAffinityManagerBase(getpid())};
  // FIXME: this only works on a twelf core cpu

  auto expected_result{std::bitset<12>("101010101010")};
  manager.SetSelectedCpus(expected_result);
  auto actual_result{manager.GetSelectedCpus()};
  EXPECT_EQ(actual_result, expected_result);
};
TEST(CpuAffinityManagerBase, GetAffinity_Initialized_FullBitset) {
  auto manager{CpuAffinityManagerBase(getpid())};
  // FIXME: this only works on a twelf core cpu

  auto expected_result{std::bitset<12>("111111111111")};
  auto actual_result{manager.GetAffinity()};
  EXPECT_EQ(actual_result, expected_result);
};
TEST(CpuAffinityManagerBase, ApplyAffinity_Set_ValidBitset) {
  auto manager{CpuAffinityManagerBase(getpid())};
  // FIXME: this only works on a twelf core cpu

  auto expected_result{std::bitset<12>("101010101010")};
  manager.SetSelectedCpus(expected_result);
  manager.ApplyAffinity();
  auto actual_result{manager.GetAffinity()};
  EXPECT_EQ(actual_result, expected_result);
};