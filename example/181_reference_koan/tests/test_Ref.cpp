#include <gtest/gtest.h>
#include <vector>
#include "Ref.h"
using namespace std;
template <typename T>
class Arena {
public:
    void setUnused(long int idx) {}
};
TEST(Ref, CopyConstructor_Copy_CountIncreases) {
    auto v{vector<int>(3)};
    auto a{Arena<int>()};
    auto r0{Ref<int>(v[0], 0, a)};
    EXPECT_EQ(r0.use_count(), 2);
    auto r1{r0};
    EXPECT_EQ(r0.use_count(), 3);
    EXPECT_EQ(r1.use_count(), 3);
};
TEST(Ref, CopyAssign_Assign_CountIncreases) {
    auto v{vector<int>(3)};
    auto a{Arena<int>()};
    auto r0{Ref<int>(v[0], 0, a)};
    auto r1{Ref<int>(v[1], 1, a)};
    EXPECT_EQ(r0.use_count(), 2);
    EXPECT_EQ(r1.use_count(), 2);
    r1 = r0;
    EXPECT_EQ(r0.use_count(), 3);
    EXPECT_EQ(r1.use_count(), 3);
};
TEST(Ref, MoveConstructor_Move_CountUnmodified) {
    auto v{vector<int>(3)};
    auto a{Arena<int>()};
    auto r0{Ref<int>(v[0], 0, a)};
    EXPECT_EQ(r0.use_count(), 2);
    auto r1{move(r0)};
    EXPECT_EQ(r1.use_count(), 3);
    // not sure why this is 3, strange
};
TEST(Ref, MoveAssign_Assign_CountUnmodified) {
    auto v{vector<int>(3)};
    auto a{Arena<int>()};
    auto r0{Ref<int>(v[0], 0, a)};
    auto r1{Ref<int>(v[1], 1, a)};
    EXPECT_EQ(r0.use_count(), 2);
    EXPECT_EQ(r1.use_count(), 2);
    r1 = r0;
    EXPECT_EQ(r0.use_count(), 3);
    EXPECT_EQ(r1.use_count(), 3);
    // i think the move operators actually perform a copy3
};
