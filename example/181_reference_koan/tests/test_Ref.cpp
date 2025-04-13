#include "Ref.h"
#include <gtest/gtest.h>
#include <vector>
using namespace std;
template <typename T> class Arena {
public:
  void setUnused(int idx) {}
};
TEST(Ref, CopyConstructor_Copy_CountIncreases) {
  auto v{vector<int>(3)};
  auto a{Arena<int>()};
  auto r0{Ref<int>((v)[(0)], 0, a)};
};