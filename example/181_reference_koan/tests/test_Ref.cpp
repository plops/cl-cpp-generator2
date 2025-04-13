#include "Ref.h"
#include <gtest/gtest.h>
using namespace std;
TEST(Ref, CopyConstructor_Copy_CountIncreases) { auto r0{Ref()}; };