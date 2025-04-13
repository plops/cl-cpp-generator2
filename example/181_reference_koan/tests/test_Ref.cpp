#include "Ref.h"
#include <gtest/gtest.h>
#include <vector>
using namespace std;
template <typename T> class Arena {
public:
  void setUnused(long int idx) {}
};
TEST(Ref, CopyConstructor_Copy_CountIncreases) {
  auto v{vector<int>(3)};
  auto a{Arena<int>()};
  auto r0{Ref<int>((v)[(0)], 0, a)};
  EXPECT_EQ(r0.use_count(), 2);
  auto r1{r0};
  EXPECT_EQ(r0.use_count(), 3);
  EXPECT_EQ(r1.use_count(), 3);
};