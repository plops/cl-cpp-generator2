#include "CpuAffinityManagerBase.h"
#include <gtest/gtest.h>
#include <thread>
#include <unistd.h>
class CpuAffinityManagerBaseTest : public ::testing::Test {
public:
  CpuAffinityManagerBaseTest()
      : n(std::thread::hardware_concurrency()), pid(getpid()),
        manager(CpuAffinityManagerBase(pid, n)) {}

protected:
  void SetUp() {}
  void TearDown() {}
  int n, pid;
  CpuAffinityManagerBase manager;
};
TEST_F(CpuAffinityManagerBaseTest, GetSelectedCpus_Initialized_FullBitset) {
  auto expected_result{std::vector<bool>(n, true)};
  auto actual_result{manager.GetSelectedCpus()};
  EXPECT_EQ(actual_result, expected_result);
};
TEST_F(CpuAffinityManagerBaseTest, SetSelectedCpus_Set_ValidBitset) {
  auto expected_result{std::vector<bool>(n, true)};
  expected_result[0] = false;
  manager.SetSelectedCpus(expected_result);
  auto actual_result{manager.GetSelectedCpus()};
  EXPECT_EQ(actual_result, expected_result);
};
TEST_F(CpuAffinityManagerBaseTest, GetAffinity_Initialized_FullBitset) {
  auto expected_result{std::vector<bool>(n, true)};
  auto actual_result{manager.GetAffinity()};
  EXPECT_EQ(actual_result, expected_result);
};
TEST_F(CpuAffinityManagerBaseTest, ApplyAffinity_Set_ValidBitset) {
  auto manager{
      CpuAffinityManagerBase(getpid(), std::thread::hardware_concurrency())};
  auto expected_result{std::vector<bool>(n, true)};
  expected_result[0] = false;
  manager.SetSelectedCpus(expected_result);
  manager.ApplyAffinity();
  auto actual_result{manager.GetAffinity()};
  EXPECT_EQ(actual_result, expected_result);
};